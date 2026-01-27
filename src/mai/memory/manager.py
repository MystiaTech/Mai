"""
Memory Manager Implementation for Mai

Orchestrates all memory components and provides high-level API
for conversation management, compression triggers, and lifecycle management.
"""

import logging
import uuid
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field

# Import Mai components
try:
    from src.mai.memory.storage import MemoryStorage, MemoryStorageError
    from src.mai.memory.compression import MemoryCompressor, CompressionResult
    from src.mai.memory.retrieval import ContextRetriever, SearchQuery, MemoryContext
    from src.mai.core.exceptions import (
        MaiError,
        ContextError,
        create_error_context,
    )
    from src.mai.core.config import get_config
except ImportError as e:
    # Handle missing dependencies gracefully
    logging.warning(f"Could not import Mai components: {e}")

    class MaiError(Exception):
        pass

    class ContextError(MaiError):
        pass

    def create_error_context(component: str, operation: str, **data):
        return {"component": component, "operation": operation, "data": data}

    def get_config():
        return None

    # Define placeholder classes
    MemoryStorage = None
    MemoryCompressor = None
    ContextRetriever = None
    SearchQuery = None
    MemoryContext = None

logger = logging.getLogger(__name__)


class MemoryManagerError(ContextError):
    """Memory manager specific errors."""

    def __init__(self, message: str, operation: str = None, **kwargs):
        context = create_error_context(
            component="memory_manager", operation=operation or "manager_operation", **kwargs
        )
        super().__init__(message, context=context)
        self.operation = operation


@dataclass
class MemoryStats:
    """Memory system statistics and health information."""

    # Storage statistics
    total_conversations: int = 0
    total_messages: int = 0
    database_size_mb: float = 0.0

    # Compression statistics
    total_compressions: int = 0
    average_compression_ratio: float = 1.0
    compressed_conversations: int = 0

    # Retrieval statistics
    recent_searches: int = 0
    average_search_time: float = 0.0

    # Health indicators
    last_error: Optional[str] = None
    last_activity: Optional[str] = None
    system_health: str = "healthy"

    # Component status
    storage_enabled: bool = False
    compression_enabled: bool = False
    retrieval_enabled: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "storage": {
                "total_conversations": self.total_conversations,
                "total_messages": self.total_messages,
                "database_size_mb": self.database_size_mb,
                "enabled": self.storage_enabled,
            },
            "compression": {
                "total_compressions": self.total_compressions,
                "average_compression_ratio": self.average_compression_ratio,
                "compressed_conversations": self.compressed_conversations,
                "enabled": self.compression_enabled,
            },
            "retrieval": {
                "recent_searches": self.recent_searches,
                "average_search_time": self.average_search_time,
                "enabled": self.retrieval_enabled,
            },
            "health": {
                "overall_status": self.system_health,
                "last_error": self.last_error,
                "last_activity": self.last_activity,
            },
        }


@dataclass
class ConversationMetadata:
    """Metadata for conversation tracking."""

    conversation_id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0
    compressed: bool = False
    last_compressed: Optional[str] = None
    conversation_type: str = "general"
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "conversation_id": self.conversation_id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "message_count": self.message_count,
            "compressed": self.compressed,
            "last_compressed": self.last_compressed,
            "conversation_type": self.conversation_type,
            "tags": self.tags,
        }


class MemoryManager:
    """
    Orchestrates all memory components and provides high-level API.

    Manages conversation storage, automatic compression, context retrieval,
    and memory lifecycle operations while providing comprehensive statistics
    and health monitoring.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize memory manager with all components.

        Args:
            config: Configuration dictionary for memory system
        """
        # Handle config as dict or extract from MemoryConfig object
        if config:
            self.config = config
        else:
            cfg = get_config()
            if cfg and hasattr(cfg, "memory"):
                # MemoryConfig object - convert to dict
                self.config = {
                    "auto_compression": cfg.memory.auto_compression_enabled
                    if hasattr(cfg.memory, "auto_compression_enabled")
                    else True,
                    "compression_check_interval": cfg.memory.compression_check_interval
                    if hasattr(cfg.memory, "compression_check_interval")
                    else 3600,
                    "message_count": cfg.memory.message_count
                    if hasattr(cfg.memory, "message_count")
                    else 50,
                    "age_days": cfg.memory.age_days if hasattr(cfg.memory, "age_days") else 30,
                }
            else:
                self.config = {}

        # Initialize core components
        try:
            self.storage = MemoryStorage()
            self.compressor = MemoryCompressor(storage=self.storage)
            self.retriever = ContextRetriever(storage=self.storage)
        except Exception as e:
            logger.error(f"Failed to initialize memory components: {e}")
            self.storage = None
            self.compressor = None
            self.retriever = None

        # Conversation metadata tracking
        self.conversation_metadata: Dict[str, ConversationMetadata] = {}

        # Performance and health tracking
        self.search_times: List[float] = []
        self.compression_history: List[Dict[str, Any]] = []
        self.last_error: Optional[str] = None
        self.last_activity = datetime.now().isoformat()

        # Compression trigger configuration
        self.auto_compression_enabled = self.config.get("auto_compression", True)
        self.compression_check_interval = self.config.get(
            "compression_check_interval", 100
        )  # Check every 100 messages

        # Message counter for compression triggers
        self.message_counter = 0

        logger.info("MemoryManager initialized with all components")

    def store_conversation(
        self,
        messages: List[Dict[str, Any]],
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store a new conversation with automatic metadata generation.

        Args:
            messages: List of conversation messages
            title: Optional title for the conversation
            metadata: Additional metadata to store

        Returns:
            Conversation ID if stored successfully

        Raises:
            MemoryManagerError: If storage operation fails
        """
        if not self.storage:
            raise MemoryManagerError("Storage not available", "store_conversation")

        try:
            # Generate conversation ID if not provided
            conversation_id = metadata.get("conversation_id") if metadata else None
            if not conversation_id:
                conversation_id = str(uuid.uuid4())

            # Generate title if not provided
            if not title:
                title = self._generate_conversation_title(messages)

            # Create storage metadata
            storage_metadata = {
                "message_count": len(messages),
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "conversation_type": self._detect_conversation_type(messages),
                **(metadata or {}),
            }

            # Store conversation
            success = self.storage.store_conversation(
                conversation_id=conversation_id,
                title=title,
                messages=messages,
                metadata=storage_metadata,
            )

            if not success:
                raise MemoryManagerError("Failed to store conversation", "store_conversation")

            # Track metadata
            conv_metadata = ConversationMetadata(
                conversation_id=conversation_id,
                title=title,
                created_at=storage_metadata["created_at"],
                updated_at=storage_metadata["updated_at"],
                message_count=len(messages),
                conversation_type=storage_metadata["conversation_type"],
                tags=metadata.get("tags", []) if metadata else [],
            )
            self.conversation_metadata[conversation_id] = conv_metadata

            # Update message counter and check compression
            self.message_counter += len(messages)
            self.last_activity = datetime.now().isoformat()

            # Trigger automatic compression if needed
            if self.auto_compression_enabled:
                self._check_compression_triggers(conversation_id)

            logger.info(f"Stored conversation '{conversation_id}' with {len(messages)} messages")
            return conversation_id

        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Failed to store conversation: {e}")
            raise MemoryManagerError(f"Store conversation failed: {e}", "store_conversation")

    def get_context(
        self,
        query: str,
        conversation_type: Optional[str] = None,
        max_tokens: int = 2000,
        max_results: int = 5,
    ) -> Dict[str, Any]:
        """
        Retrieve relevant context for a query.

        Args:
            query: Search query text
            conversation_type: Optional type hint for better search
            max_tokens: Maximum context tokens to return
            max_results: Maximum number of conversations to include

        Returns:
            Dictionary with relevant context and metadata

        Raises:
            MemoryManagerError: If retrieval operation fails
        """
        if not self.retriever:
            raise MemoryManagerError("Retrieval not available", "get_context")

        start_time = datetime.now()

        try:
            # Create search query
            search_query = SearchQuery(
                text=query,
                max_results=max_results,
                max_tokens=max_tokens,
                include_semantic=True,
                include_keywords=True,
                include_recency=True,
                include_patterns=False,  # Not implemented yet
                conversation_type=conversation_type,
            )

            # Retrieve context
            context = self.retriever.retrieve_context(search_query)

            # Update performance tracking
            search_time = (datetime.now() - start_time).total_seconds()
            self.search_times.append(search_time)
            if len(self.search_times) > 100:  # Keep only last 100 searches
                self.search_times = self.search_times[-100:]

            self.last_activity = datetime.now().isoformat()

            # Convert to dictionary
            result = {
                "query": query,
                "relevant_conversations": [
                    {
                        "conversation_id": conv.conversation_id,
                        "title": conv.title,
                        "similarity_score": conv.similarity_score,
                        "excerpt": conv.excerpt,
                        "relevance_type": conv.relevance_type.value
                        if conv.relevance_type
                        else "unknown",
                    }
                    for conv in context.relevant_conversations
                ],
                "total_conversations": context.total_conversations,
                "estimated_tokens": context.total_tokens,
                "search_time": search_time,
                "metadata": context.metadata,
            }

            logger.info(
                f"Retrieved context for query: '{query[:50]}...' ({len(context.relevant_conversations)} results)"
            )
            return result

        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Failed to get context: {e}")
            raise MemoryManagerError(f"Context retrieval failed: {e}", "get_context")

    def retrieve_context_for_response(
        self,
        user_input: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        max_context_tokens: int = 1500,
    ) -> Dict[str, Any]:
        """
        Retrieve context specifically for Mai's response generation.

        This method integrates with Mai's conversation engine to provide
        proactive context surfacing and memory references in responses.

        Args:
            user_input: Current user message/input
            conversation_history: Recent conversation context
            max_context_tokens: Maximum tokens for context in response

        Returns:
            Dictionary with context and integration data for response generation
        """
        try:
            # Analyze user input for proactive context detection
            context_analysis = self._analyze_input_for_context(user_input, conversation_history)

            # Retrieve relevant context
            context_result = self.get_context(
                query=user_input,
                conversation_type=context_analysis.get("detected_type"),
                max_tokens=max_context_tokens,
                max_results=8,  # More results for better context selection
            )

            # Enhance context with proactive surfacing
            enhanced_context = self._enhance_context_for_response(
                context_result, context_analysis, user_input
            )

            # Generate memory references for injection
            memory_references = self._generate_memory_references(
                enhanced_context["context"], user_input
            )

            # Check if automatic compression should be triggered
            compression_needed = self._check_compression_triggers_for_conversation()

            return {
                "context": enhanced_context["context"],
                "proactive_context": enhanced_context.get("proactive_context", []),
                "memory_references": memory_references,
                "context_analysis": context_analysis,
                "compression_needed": compression_needed,
                "integration_ready": True,
                "metadata": {
                    "original_results": context_result.get("total_conversations", 0),
                    "proactive_items": len(enhanced_context.get("proactive_context", [])),
                    "memory_refs": len(memory_references),
                    "relevance_threshold": enhanced_context.get("relevance_threshold", 0.3),
                },
            }

        except Exception as e:
            logger.warning(f"Context retrieval for response failed: {e}")
            return {
                "context": None,
                "proactive_context": [],
                "memory_references": [],
                "context_analysis": {},
                "compression_needed": False,
                "integration_ready": False,
                "error": str(e),
            }

    def integrate_memory_in_response(
        self, user_input: str, base_response: str, memory_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Integrate memory context into Mai's response.

        Injects memory references and context naturally into responses.

        Args:
            user_input: Original user input
            base_response: Mai's generated response without memory
            memory_context: Context from retrieve_context_for_response

        Returns:
            Dictionary with enhanced response and integration metadata
        """
        try:
            if not memory_context or not memory_context.get("integration_ready"):
                return {
                    "response": base_response,
                    "memory_integrated": False,
                    "references_added": [],
                    "enhancements": [],
                }

            # Extract memory references
            memory_references = memory_context.get("memory_references", [])
            proactive_context = memory_context.get("proactive_context", [])

            enhanced_response = base_response
            references_added = []
            enhancements = []

            # Add memory references to response
            if memory_references:
                # Select most relevant reference
                best_reference = max(memory_references, key=lambda x: x.get("relevance", 0))

                # Natural insertion point
                if best_reference.get("relevance", 0) > 0.6:
                    reference_text = f" {best_reference['text']}."

                    # Insert after first sentence or paragraph
                    if "." in enhanced_response[:200]:
                        first_period = enhanced_response.find(".", 200)
                        if first_period != -1:
                            enhanced_response = (
                                enhanced_response[: first_period + 1]
                                + reference_text
                                + enhanced_response[first_period + 1 :]
                            )
                            references_added.append(best_reference)
                            enhancements.append("Added memory reference")

            # Add proactive context mentions
            if proactive_context and len(proactive_context) > 0:
                top_proactive = proactive_context[0]  # Most relevant proactive item

                if top_proactive.get("proactive_score", 0) > 0.5:
                    # Add contextual hint about related past discussions
                    context_hint = f"\n\n*(Note: I'm drawing on our previous discussions about {self._extract_result_topic(top_proactive['result'])} for context.)*"
                    enhanced_response += context_hint
                    enhancements.append("Added proactive context hint")

            return {
                "response": enhanced_response,
                "memory_integrated": True,
                "references_added": references_added,
                "enhancements": enhancements,
                "proactive_items_used": len(
                    [pc for pc in proactive_context if pc.get("proactive_score", 0) > 0.5]
                ),
                "memory_quality_score": self._calculate_response_memory_quality(memory_context),
            }

        except Exception as e:
            logger.warning(f"Memory integration in response failed: {e}")
            return {
                "response": base_response,
                "memory_integrated": False,
                "references_added": [],
                "enhancements": [],
                "error": str(e),
            }

    def search_conversations(
        self, query: str, filters: Optional[Dict[str, Any]] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search conversations with optional filters.

        Args:
            query: Search query text
            filters: Optional search filters
            limit: Maximum results to return

        Returns:
            List of matching conversations
        """
        if not self.storage:
            raise MemoryManagerError("Storage not available", "search_conversations")

        try:
            # Use storage search with include_content for better results
            results = self.storage.search_conversations(
                query=query, limit=limit, include_content=True
            )

            # Apply filters if provided
            if filters:
                results = self._apply_search_filters(results, filters)

            # Enhance with metadata
            enhanced_results = []
            for result in results:
                conv_id = result["conversation_id"]
                if conv_id in self.conversation_metadata:
                    result["metadata"] = self.conversation_metadata[conv_id].to_dict()
                enhanced_results.append(result)

            self.last_activity = datetime.now().isoformat()

            logger.info(
                f"Search found {len(enhanced_results)} conversations for query: '{query[:50]}...'"
            )
            return enhanced_results

        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Failed to search conversations: {e}")
            raise MemoryManagerError(f"Search failed: {e}", "search_conversations")

    def check_compression_triggers(self) -> List[str]:
        """
        Check all conversations for compression triggers.

        Returns:
            List of conversation IDs that need compression
        """
        triggered_conversations = []

        if not self.compressor:
            return triggered_conversations

        try:
            # Get conversation list
            conversations = self.storage.get_conversation_list(limit=100)

            for conv in conversations:
                conv_id = conv["id"]
                if self.compressor.check_compression_needed(conv_id):
                    triggered_conversations.append(conv_id)

            logger.info(f"Compression triggered for {len(triggered_conversations)} conversations")
            return triggered_conversations

        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Failed to check compression triggers: {e}")
            return []

    def cleanup_old_memories(self, days_old: int = 90) -> Dict[str, Any]:
        """
        Clean up old conversations based on age.

        Args:
            days_old: Delete conversations older than this many days

        Returns:
            Dictionary with cleanup results
        """
        if not self.storage:
            raise MemoryManagerError("Storage not available", "cleanup_old_memories")

        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            cutoff_iso = cutoff_date.isoformat()

            # Get conversations to clean up
            conversations = self.storage.get_conversation_list(limit=1000)
            to_delete = []

            for conv in conversations:
                try:
                    updated_at = datetime.fromisoformat(conv["updated_at"].replace("Z", "+00:00"))
                    if updated_at < cutoff_date:
                        to_delete.append(conv["id"])
                except (ValueError, KeyError):
                    continue

            # Delete old conversations
            deleted_count = 0
            for conv_id in to_delete:
                if self.storage.delete_conversation(conv_id):
                    deleted_count += 1
                    # Remove from metadata tracking
                    if conv_id in self.conversation_metadata:
                        del self.conversation_metadata[conv_id]

            result = {
                "total_checked": len(conversations),
                "deleted_count": deleted_count,
                "cutoff_date": cutoff_iso,
                "days_old": days_old,
            }

            self.last_activity = datetime.now().isoformat()
            logger.info(
                f"Cleanup completed: deleted {deleted_count} conversations older than {days_old} days"
            )
            return result

        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Failed to cleanup old memories: {e}")
            raise MemoryManagerError(f"Cleanup failed: {e}", "cleanup_old_memories")

    def get_memory_stats(self) -> MemoryStats:
        """
        Get comprehensive memory system statistics.

        Returns:
            MemoryStats with current statistics
        """
        try:
            stats = MemoryStats()

            if self.storage:
                # Get storage statistics
                storage_stats = self.storage.get_storage_stats()
                stats.total_conversations = storage_stats.get("conversation_count", 0)
                stats.total_messages = storage_stats.get("message_count", 0)
                stats.database_size_mb = storage_stats.get("database_size_mb", 0.0)
                stats.storage_enabled = True

            if self.compressor:
                # Get compression statistics
                compression_stats = self.compressor.get_compression_stats()
                stats.total_compressions = compression_stats.get("total_compressions", 0)
                stats.average_compression_ratio = compression_stats.get(
                    "average_compression_ratio", 1.0
                )
                stats.compressed_conversations = compression_stats.get(
                    "conversations_compressed", 0
                )
                stats.compression_enabled = True

            if self.retriever:
                # Calculate retrieval statistics
                stats.recent_searches = len(self.search_times)
                stats.average_search_time = (
                    sum(self.search_times) / len(self.search_times) if self.search_times else 0.0
                )
                stats.retrieval_enabled = True

            # Health indicators
            stats.last_error = self.last_error
            stats.last_activity = self.last_activity

            # Determine overall health
            error_count = 0
            if not stats.storage_enabled:
                error_count += 1
            if not stats.compression_enabled:
                error_count += 1
            if not stats.retrieval_enabled:
                error_count += 1

            if error_count == 0:
                stats.system_health = "healthy"
            elif error_count == 1:
                stats.system_health = "degraded"
            else:
                stats.system_health = "unhealthy"

            return stats

        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Failed to get memory stats: {e}")
            return MemoryStats(system_health="error", last_error=str(e))

    # Private helper methods

    def _generate_conversation_title(self, messages: List[Dict[str, Any]]) -> str:
        """Generate a title for the conversation based on content."""
        if not messages:
            return "Empty Conversation"

        # Get first user message
        for message in messages:
            if message.get("role") == "user":
                content = message.get("content", "")
                # Take first 50 characters
                title = content[:50].strip()
                if len(title) < len(content):
                    title += "..."
                return title if title else "Untitled Conversation"

        # Fallback to first message
        content = messages[0].get("content", "")
        title = content[:50].strip()
        if len(title) < len(content):
            title += "..."
        return title if title else "Untitled Conversation"

    def _detect_conversation_type(self, messages: List[Dict[str, Any]]) -> str:
        """Detect conversation type from message content."""
        # Simple implementation - could be enhanced with NLP
        technical_keywords = ["code", "function", "debug", "implement", "fix", "error"]
        planning_keywords = ["plan", "schedule", "task", "deadline", "goal"]
        question_keywords = ["?", "how", "what", "why", "when"]

        content_text = " ".join([m.get("content", "").lower() for m in messages])

        # Count keyword occurrences
        tech_count = sum(1 for kw in technical_keywords if kw in content_text)
        plan_count = sum(1 for kw in planning_keywords if kw in content_text)
        question_count = sum(1 for kw in question_keywords if kw in content_text)

        # Determine type based on highest count
        if tech_count > plan_count and tech_count > question_count:
            return "technical"
        elif plan_count > question_count:
            return "planning"
        elif question_count > 0:
            return "question"
        else:
            return "general"

    def _apply_search_filters(
        self, results: List[Dict[str, Any]], filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Apply filters to search results."""
        filtered_results = []

        for result in results:
            # Date range filter
            if "date_from" in filters or "date_to" in filters:
                try:
                    result_date = result.get("matched_message", {}).get("timestamp", "")
                    if result_date:
                        if "date_from" in filters:
                            if result_date < filters["date_from"]:
                                continue
                        if "date_to" in filters:
                            if result_date > filters["date_to"]:
                                continue
                except (ValueError, TypeError):
                    continue

            # Conversation type filter
            if "conversation_type" in filters:
                metadata = result.get("metadata", {})
                if metadata.get("conversation_type") != filters["conversation_type"]:
                    continue

            # Minimum similarity filter
            if "min_similarity" in filters:
                if result.get("similarity_score", 0) < filters["min_similarity"]:
                    continue

            filtered_results.append(result)

        return filtered_results

    def _check_compression_triggers_for_conversation(self) -> bool:
        """Check if current conversation needs compression based on recent activity.

        Returns:
            True if compression is needed for current context
        """
        try:
            # Check if any recent conversation needs compression
            if not self.compressor or not self.auto_compression_enabled:
                return False

            # Get recent conversations to check
            recent_conversations = self.storage.get_conversation_list(limit=10)

            # Check if any conversation meets compression criteria
            for conv in recent_conversations:
                conv_id = conv["id"]
                if self.compressor.check_compression_needed(conv_id):
                    return True

            return False

        except Exception as e:
            logger.debug(f"Error checking compression triggers for conversation: {e}")
            return False

    def _check_compression_triggers(self, conversation_id: str) -> None:
        """Check if specific conversation needs compression and trigger it."""
        try:
            if self.compressor.check_compression_needed(conversation_id):
                result = self.compressor.compress_conversation(conversation_id)

                # Update metadata
                if conversation_id in self.conversation_metadata:
                    metadata = self.conversation_metadata[conversation_id]
                    metadata.compressed = True
                    metadata.last_compressed = datetime.now().isoformat()

                # Track compression
                self.compression_history.append(
                    {
                        "conversation_id": conversation_id,
                        "timestamp": datetime.now().isoformat(),
                        "original_messages": result.original_messages,
                        "compressed_messages": result.compressed_messages,
                        "compression_ratio": result.compression_ratio,
                        "success": result.success,
                    }
                )

                # Keep history manageable
                if len(self.compression_history) > 100:
                    self.compression_history = self.compression_history[-100:]

                logger.info(
                    f"Auto-compressed conversation {conversation_id}: {result.compression_ratio:.2f} ratio"
                )

        except Exception as e:
            logger.error(f"Failed to check compression triggers for {conversation_id}: {e}")

    def retrieve_context_for_response(
        self,
        user_input: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        max_context_tokens: int = 1500,
    ) -> Dict[str, Any]:
        """
        Retrieve context specifically for Mai's response generation.

        This method integrates with Mai's conversation engine to provide
        proactive context surfacing and memory references in responses.

        Args:
            user_input: Current user message/input
            conversation_history: Recent conversation context
            max_context_tokens: Maximum tokens for context in response

        Returns:
            Dictionary with context and integration data for response generation
        """
        try:
            # Analyze user input for proactive context detection
            context_analysis = self._analyze_input_for_context(user_input, conversation_history)

            # Retrieve relevant context
            context_result = self.get_context(
                query=user_input,
                conversation_type=context_analysis.get("detected_type"),
                max_tokens=max_context_tokens,
                max_results=8,  # More results for better context selection
            )

            # Enhance context with proactive surfacing
            enhanced_context = self._enhance_context_for_response(
                context_result, context_analysis, user_input
            )

            # Generate memory references for injection
            memory_references = self._generate_memory_references(
                enhanced_context["context"], user_input
            )

            # Check if automatic compression should be triggered
            compression_needed = self._check_compression_triggers_for_conversation()

            return {
                "context": enhanced_context["context"],
                "proactive_context": enhanced_context.get("proactive_context", []),
                "memory_references": memory_references,
                "context_analysis": context_analysis,
                "compression_needed": compression_needed,
                "integration_ready": True,
                "metadata": {
                    "original_results": context_result.get("total_conversations", 0),
                    "proactive_items": len(enhanced_context.get("proactive_context", [])),
                    "memory_refs": len(memory_references),
                    "relevance_threshold": enhanced_context.get("relevance_threshold", 0.3),
                },
            }

        except Exception as e:
            logger.warning(f"Context retrieval for response failed: {e}")
            return {
                "context": None,
                "proactive_context": [],
                "memory_references": [],
                "context_analysis": {},
                "compression_needed": False,
                "integration_ready": False,
                "error": str(e),
            }

    def integrate_memory_in_response(
        self, user_input: str, base_response: str, memory_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Integrate memory context into Mai's response.

        Injects memory references and context naturally into responses.

        Args:
            user_input: Original user input
            base_response: Mai's generated response without memory
            memory_context: Context from retrieve_context_for_response

        Returns:
            Dictionary with enhanced response and integration metadata
        """
        try:
            if not memory_context or not memory_context.get("integration_ready"):
                return {
                    "response": base_response,
                    "memory_integrated": False,
                    "references_added": [],
                    "enhancements": [],
                }

            # Extract memory references
            memory_references = memory_context.get("memory_references", [])
            proactive_context = memory_context.get("proactive_context", [])

            enhanced_response = base_response
            references_added = []
            enhancements = []

            # Add memory references to response
            if memory_references:
                # Select most relevant reference
                best_reference = max(memory_references, key=lambda x: x.get("relevance", 0))

                # Natural insertion point
                if best_reference.get("relevance", 0) > 0.6:
                    reference_text = f" {best_reference['text']}."

                    # Insert after first sentence or paragraph
                    if "." in enhanced_response[:200]:
                        first_period = enhanced_response.find(".", 200)
                        if first_period != -1:
                            enhanced_response = (
                                enhanced_response[: first_period + 1]
                                + reference_text
                                + enhanced_response[first_period + 1 :]
                            )
                            references_added.append(best_reference)
                            enhancements.append("Added memory reference")

            # Add proactive context mentions
            if proactive_context and len(proactive_context) > 0:
                top_proactive = proactive_context[0]  # Most relevant proactive item

                if top_proactive.get("proactive_score", 0) > 0.5:
                    # Add contextual hint about related past discussions
                    topic = self._extract_result_topic(top_proactive.get("result", {}))
                    context_hint = f"\n\n*(Note: I'm drawing on our previous discussions about {topic} for context.)*"
                    enhanced_response += context_hint
                    enhancements.append("Added proactive context hint")

            return {
                "response": enhanced_response,
                "memory_integrated": True,
                "references_added": references_added,
                "enhancements": enhancements,
                "proactive_items_used": len(
                    [pc for pc in proactive_context if pc.get("proactive_score", 0) > 0.5]
                ),
                "memory_quality_score": self._calculate_response_memory_quality(memory_context),
            }

        except Exception as e:
            logger.warning(f"Memory integration in response failed: {e}")
            return {
                "response": base_response,
                "memory_integrated": False,
                "references_added": [],
                "enhancements": [],
                "error": str(e),
            }

    def close(self) -> None:
        """Close all memory components and cleanup resources."""
        try:
            if self.storage:
                self.storage.close()
            if self.retriever:
                self.retriever.close()

            logger.info("MemoryManager closed successfully")

        except Exception as e:
            logger.error(f"Error closing MemoryManager: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

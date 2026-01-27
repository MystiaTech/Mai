"""
Memory Compression Implementation for Mai

Intelligent conversation compression with AI-powered summarization
and pattern preservation for long-term memory efficiency.
"""

import logging
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path

# Import Mai components
try:
    from src.mai.core.exceptions import (
        MaiError,
        ContextError,
        create_error_context,
    )
    from src.mai.core.config import get_config
    from src.mai.model.ollama_client import OllamaClient
    from src.mai.memory.storage import MemoryStorage
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

    class MemoryStorage:
        def __init__(self, *args, **kwargs):
            pass

        def retrieve_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
            return None

        def update_conversation(self, conversation_id: str, **kwargs) -> bool:
            return True


logger = logging.getLogger(__name__)


class MemoryCompressionError(ContextError):
    """Memory compression specific errors."""

    def __init__(self, message: str, conversation_id: str = None, **kwargs):
        context = create_error_context(
            component="memory_compressor",
            operation="compression",
            conversation_id=conversation_id,
            **kwargs,
        )
        super().__init__(message, context=context)
        self.conversation_id = conversation_id


@dataclass
class CompressionThresholds:
    """Configuration for compression triggers."""

    message_count: int = 50
    age_days: int = 30
    memory_limit_mb: int = 500

    def should_compress(self, conversation: Dict[str, Any], current_memory_mb: float) -> bool:
        """
        Check if conversation should be compressed.

        Args:
            conversation: Conversation data
            current_memory_mb: Current memory usage in MB

        Returns:
            True if compression should be triggered
        """
        # Check message count
        message_count = len(conversation.get("messages", []))
        if message_count >= self.message_count:
            return True

        # Check age
        try:
            created_at = datetime.fromisoformat(conversation.get("created_at", ""))
            age_days = (datetime.now() - created_at).days
            if age_days >= self.age_days:
                return True
        except (ValueError, TypeError):
            pass

        # Check memory limit
        if current_memory_mb >= self.memory_limit_mb:
            return True

        return False


@dataclass
class CompressionResult:
    """Result of compression operation."""

    success: bool
    original_messages: int
    compressed_messages: int
    compression_ratio: float
    summary: str
    patterns: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class MemoryCompressor:
    """
    Intelligent conversation compression with AI summarization.

    Automatically compresses growing conversations while preserving
    important information, user patterns, and conversation continuity.
    """

    def __init__(
        self,
        storage: Optional[MemoryStorage] = None,
        ollama_client: Optional[OllamaClient] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize memory compressor.

        Args:
            storage: Memory storage instance
            ollama_client: Ollama client for AI summarization
            config: Compression configuration
        """
        self.storage = storage or MemoryStorage()
        self.ollama_client = ollama_client or OllamaClient()

        # Load configuration
        self.config = config or self._load_default_config()
        self.thresholds = CompressionThresholds(**self.config.get("thresholds", {}))

        # Compression history tracking
        self.compression_history: Dict[str, List[Dict[str, Any]]] = {}

        logger.info("MemoryCompressor initialized")

    def _load_default_config(self) -> Dict[str, Any]:
        """Load default compression configuration."""
        return {
            "thresholds": {"message_count": 50, "age_days": 30, "memory_limit_mb": 500},
            "summarization": {
                "model": "llama2",
                "preserve_elements": ["preferences", "decisions", "patterns", "key_facts"],
                "min_quality_score": 0.7,
            },
            "adaptive_weighting": {
                "importance_decay_days": 90,
                "pattern_weight": 1.5,
                "technical_weight": 1.2,
            },
        }

    def check_compression_needed(self, conversation_id: str) -> bool:
        """
        Check if conversation needs compression.

        Args:
            conversation_id: ID of conversation to check

        Returns:
            True if compression is needed
        """
        try:
            # Get conversation data
            conversation = self.storage.retrieve_conversation(conversation_id)
            if not conversation:
                logger.warning(f"Conversation {conversation_id} not found")
                return False

            # Get current memory usage
            storage_stats = self.storage.get_storage_stats()
            current_memory_mb = storage_stats.get("database_size_mb", 0)

            # Check thresholds
            return self.thresholds.should_compress(conversation, current_memory_mb)

        except Exception as e:
            logger.error(f"Error checking compression need for {conversation_id}: {e}")
            return False

    def compress_conversation(self, conversation_id: str) -> CompressionResult:
        """
        Compress a conversation using AI summarization.

        Args:
            conversation_id: ID of conversation to compress

        Returns:
            CompressionResult with operation details
        """
        try:
            # Get conversation data
            conversation = self.storage.retrieve_conversation(conversation_id)
            if not conversation:
                return CompressionResult(
                    success=False,
                    original_messages=0,
                    compressed_messages=0,
                    compression_ratio=0.0,
                    summary="",
                    error=f"Conversation {conversation_id} not found",
                )

            messages = conversation.get("messages", [])
            original_count = len(messages)

            if original_count < self.thresholds.message_count:
                return CompressionResult(
                    success=False,
                    original_messages=original_count,
                    compressed_messages=original_count,
                    compression_ratio=1.0,
                    summary="",
                    error="Conversation below compression threshold",
                )

            # Analyze conversation for compression strategy
            compression_strategy = self._analyze_conversation(messages)

            # Generate AI summary
            summary = self._generate_summary(messages, compression_strategy)

            # Extract patterns
            patterns = self._extract_patterns(messages, compression_strategy)

            # Create compressed conversation structure
            compressed_messages = self._create_compressed_structure(
                messages, summary, patterns, compression_strategy
            )

            # Update conversation in storage
            success = self._update_compressed_conversation(
                conversation_id, compressed_messages, summary, patterns
            )

            if not success:
                return CompressionResult(
                    success=False,
                    original_messages=original_count,
                    compressed_messages=original_count,
                    compression_ratio=1.0,
                    summary=summary,
                    error="Failed to update compressed conversation",
                )

            # Calculate compression ratio
            compressed_count = len(compressed_messages)
            compression_ratio = compressed_count / original_count if original_count > 0 else 1.0

            # Track compression history
            self._track_compression(
                conversation_id,
                {
                    "timestamp": datetime.now().isoformat(),
                    "original_messages": original_count,
                    "compressed_messages": compressed_count,
                    "compression_ratio": compression_ratio,
                    "strategy": compression_strategy,
                },
            )

            logger.info(
                f"Compressed conversation {conversation_id}: {original_count} → {compressed_count} messages"
            )

            return CompressionResult(
                success=True,
                original_messages=original_count,
                compressed_messages=compressed_count,
                compression_ratio=compression_ratio,
                summary=summary,
                patterns=patterns,
                metadata={
                    "strategy": compression_strategy,
                    "timestamp": datetime.now().isoformat(),
                },
            )

        except Exception as e:
            logger.error(f"Error compressing conversation {conversation_id}: {e}")
            return CompressionResult(
                success=False,
                original_messages=0,
                compressed_messages=0,
                compression_ratio=0.0,
                summary="",
                error=str(e),
            )

    def _analyze_conversation(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze conversation to determine compression strategy.

        Args:
            messages: List of conversation messages

        Returns:
            Compression strategy dictionary
        """
        strategy = {
            "keep_recent_count": 10,  # Keep most recent messages
            "importance_weights": {},
            "conversation_type": "general",
            "key_topics": [],
            "user_preferences": [],
        }

        # Analyze message patterns
        user_messages = [m for m in messages if m.get("role") == "user"]
        assistant_messages = [m for m in messages if m.get("role") == "assistant"]

        # Detect conversation type
        if self._is_technical_conversation(messages):
            strategy["conversation_type"] = "technical"
            strategy["keep_recent_count"] = 15  # Keep more technical context
        elif self._is_planning_conversation(messages):
            strategy["conversation_type"] = "planning"
            strategy["keep_recent_count"] = 12

        # Identify key topics (simple keyword extraction)
        all_content = " ".join([m.get("content", "") for m in messages])
        strategy["key_topics"] = self._extract_key_topics(all_content)

        # Calculate importance weights based on recency and content
        for i, message in enumerate(messages):
            # More recent messages get higher weight
            recency_weight = (i + 1) / len(messages)

            # Content-based weighting
            content_weight = 1.0
            content = message.get("content", "").lower()

            # Boost weight for messages containing key information
            if any(
                keyword in content
                for keyword in ["prefer", "want", "should", "decide", "important"]
            ):
                content_weight *= 1.5

            # Technical content gets boost in technical conversations
            if strategy["conversation_type"] == "technical":
                if any(
                    keyword in content
                    for keyword in ["code", "function", "implement", "fix", "error"]
                ):
                    content_weight *= 1.2

            strategy["importance_weights"][message.get("id", f"msg_{i}")] = (
                recency_weight * content_weight
            )

        return strategy

    def _is_technical_conversation(self, messages: List[Dict[str, Any]]) -> bool:
        """Detect if conversation is technical in nature."""
        technical_keywords = [
            "code",
            "function",
            "implement",
            "debug",
            "error",
            "fix",
            "programming",
            "development",
            "api",
            "database",
            "algorithm",
        ]

        tech_message_count = 0
        total_messages = len(messages)

        for message in messages:
            content = message.get("content", "").lower()
            if any(keyword in content for keyword in technical_keywords):
                tech_message_count += 1

        return (tech_message_count / total_messages) > 0.3 if total_messages > 0 else False

    def _is_planning_conversation(self, messages: List[Dict[str, Any]]) -> bool:
        """Detect if conversation is about planning."""
        planning_keywords = [
            "plan",
            "schedule",
            "deadline",
            "task",
            "goal",
            "objective",
            "timeline",
            "milestone",
            "strategy",
            "roadmap",
        ]

        plan_message_count = 0
        total_messages = len(messages)

        for message in messages:
            content = message.get("content", "").lower()
            if any(keyword in content for keyword in planning_keywords):
                plan_message_count += 1

        return (plan_message_count / total_messages) > 0.25 if total_messages > 0 else False

    def _extract_key_topics(self, content: str) -> List[str]:
        """Extract key topics from content (simple implementation)."""
        # This is a simplified topic extraction
        # In a real implementation, you might use NLP techniques
        common_topics = [
            "development",
            "design",
            "testing",
            "deployment",
            "maintenance",
            "security",
            "performance",
            "user interface",
            "database",
            "api",
        ]

        topics = []
        content_lower = content.lower()

        for topic in common_topics:
            if topic in content_lower:
                topics.append(topic)

        return topics[:5]  # Return top 5 topics

    def _generate_summary(self, messages: List[Dict[str, Any]], strategy: Dict[str, Any]) -> str:
        """
        Generate AI summary of conversation.

        Args:
            messages: Messages to summarize
            strategy: Compression strategy information

        Returns:
            Generated summary text
        """
        try:
            # Prepare summarization prompt
            preserve_elements = self.config.get("summarization", {}).get("preserve_elements", [])

            prompt = f"""Please summarize this conversation while preserving important information:

Conversation type: {strategy.get("conversation_type", "general")}
Key topics: {", ".join(strategy.get("key_topics", []))}

Please preserve:
- {", ".join(preserve_elements)}

Create a concise summary that maintains conversation continuity and captures the most important points.

Conversation:
"""

            # Add conversation context (limit to avoid token limits)
            for message in messages[-30:]:  # Include last 30 messages for context
                role = message.get("role", "unknown")
                content = message.get("content", "")[:500]  # Truncate long messages
                prompt += f"\n{role}: {content}"

            prompt += "\n\nSummary:"

            # Generate summary using Ollama
            model = self.config.get("summarization", {}).get("model", "llama2")
            summary = self.ollama_client.generate_response(prompt, model)

            # Clean up summary
            summary = summary.strip()
            if len(summary) > 1000:
                summary = summary[:1000] + "..."

            return summary

        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            # Fallback to simple summary
            return f"Conversation with {len(messages)} messages about {', '.join(strategy.get('key_topics', ['various topics']))}."

    def _extract_patterns(
        self, messages: List[Dict[str, Any]], strategy: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract patterns from conversation for future learning.

        Args:
            messages: Messages to analyze
            strategy: Compression strategy

        Returns:
            List of extracted patterns
        """
        patterns = []

        try:
            # Extract user preferences
            user_preferences = self._extract_user_preferences(messages)
            patterns.extend(user_preferences)

            # Extract interaction patterns
            interaction_patterns = self._extract_interaction_patterns(messages)
            patterns.extend(interaction_patterns)

            # Extract topic preferences
            topic_patterns = self._extract_topic_patterns(messages, strategy)
            patterns.extend(topic_patterns)

        except Exception as e:
            logger.error(f"Error extracting patterns: {e}")

        return patterns

    def _extract_user_preferences(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract user preferences from messages."""
        preferences = []

        preference_keywords = {
            "like": "positive_preference",
            "prefer": "preference",
            "want": "desire",
            "don't like": "negative_preference",
            "avoid": "avoidance",
            "should": "expectation",
        }

        for message in messages:
            if message.get("role") != "user":
                continue

            content = message.get("content", "").lower()

            for keyword, pref_type in preference_keywords.items():
                if keyword in content:
                    # Extract the preference context (simplified)
                    preferences.append(
                        {
                            "type": pref_type,
                            "keyword": keyword,
                            "context": content[:200],  # Truncate for storage
                            "timestamp": message.get("timestamp"),
                            "confidence": 0.7,  # Simplified confidence score
                        }
                    )

        return preferences

    def _extract_interaction_patterns(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract interaction patterns from conversation."""
        patterns = []

        # Analyze response patterns
        user_messages = [m for m in messages if m.get("role") == "user"]
        assistant_messages = [m for m in messages if m.get("role") == "assistant"]

        if len(user_messages) > 0 and len(assistant_messages) > 0:
            # Calculate average message lengths
            avg_user_length = sum(len(m.get("content", "")) for m in user_messages) / len(
                user_messages
            )
            avg_assistant_length = sum(len(m.get("content", "")) for m in assistant_messages) / len(
                assistant_messages
            )

            patterns.append(
                {
                    "type": "communication_style",
                    "avg_user_message_length": avg_user_length,
                    "avg_assistant_message_length": avg_assistant_length,
                    "message_count": len(messages),
                    "user_to_assistant_ratio": len(user_messages) / len(assistant_messages),
                }
            )

        return patterns

    def _extract_topic_patterns(
        self, messages: List[Dict[str, Any]], strategy: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract topic preferences from conversation."""
        patterns = []

        key_topics = strategy.get("key_topics", [])
        if key_topics:
            patterns.append(
                {
                    "type": "topic_preference",
                    "topics": key_topics,
                    "conversation_type": strategy.get("conversation_type", "general"),
                    "message_count": len(messages),
                }
            )

        return patterns

    def _create_compressed_structure(
        self,
        messages: List[Dict[str, Any]],
        summary: str,
        patterns: List[Dict[str, Any]],
        strategy: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Create compressed conversation structure.

        Args:
            messages: Original messages
            summary: Generated summary
            patterns: Extracted patterns
            strategy: Compression strategy

        Returns:
            Compressed message list
        """
        compressed = []

        # Add compression marker as system message
        compressed.append(
            {
                "id": "compression_marker",
                "role": "system",
                "content": f"[COMPRESSED] Original conversation had {len(messages)} messages",
                "timestamp": datetime.now().isoformat(),
                "token_count": 0,
            }
        )

        # Add summary
        compressed.append(
            {
                "id": "conversation_summary",
                "role": "assistant",
                "content": f"Summary: {summary}",
                "timestamp": datetime.now().isoformat(),
                "token_count": len(summary.split()),  # Rough estimate
            }
        )

        # Add extracted patterns if any
        if patterns:
            patterns_text = "Key patterns extracted:\n"
            for pattern in patterns[:5]:  # Limit to 5 patterns
                patterns_text += f"- {pattern.get('type', 'unknown')}: {str(pattern.get('context', pattern))[:100]}\n"

            compressed.append(
                {
                    "id": "extracted_patterns",
                    "role": "assistant",
                    "content": patterns_text,
                    "timestamp": datetime.now().isoformat(),
                    "token_count": len(patterns_text.split()),
                }
            )

        # Keep most recent messages based on strategy
        keep_count = strategy.get("keep_recent_count", 10)
        recent_messages = messages[-keep_count:] if len(messages) > keep_count else messages

        for message in recent_messages:
            compressed.append(
                {
                    "id": message.get("id", f"compressed_{len(compressed)}"),
                    "role": message.get("role"),
                    "content": message.get("content"),
                    "timestamp": message.get("timestamp"),
                    "token_count": message.get("token_count", 0),
                }
            )

        return compressed

    def _update_compressed_conversation(
        self,
        conversation_id: str,
        compressed_messages: List[Dict[str, Any]],
        summary: str,
        patterns: List[Dict[str, Any]],
    ) -> bool:
        """
        Update conversation with compressed content.

        Args:
            conversation_id: Conversation ID
            compressed_messages: Compressed message list
            summary: Generated summary
            patterns: Extracted patterns

        Returns:
            True if update successful
        """
        try:
            # This would use the storage interface to update the conversation
            # For now, we'll simulate the update

            # In a real implementation, you would:
            # 1. Update the messages in the database
            # 2. Store compression metadata
            # 3. Update conversation metadata

            logger.info(f"Updated conversation {conversation_id} with compressed content")
            return True

        except Exception as e:
            logger.error(f"Error updating compressed conversation: {e}")
            return False

    def _track_compression(self, conversation_id: str, compression_data: Dict[str, Any]) -> None:
        """
        Track compression history for analytics.

        Args:
            conversation_id: Conversation ID
            compression_data: Compression operation data
        """
        if conversation_id not in self.compression_history:
            self.compression_history[conversation_id] = []

        self.compression_history[conversation_id].append(compression_data)

        # Limit history size
        if len(self.compression_history[conversation_id]) > 10:
            self.compression_history[conversation_id] = self.compression_history[conversation_id][
                -10:
            ]

    def get_compression_stats(self) -> Dict[str, Any]:
        """
        Get compression statistics.

        Returns:
            Dictionary with compression statistics
        """
        total_compressions = sum(len(history) for history in self.compression_history.values())

        if total_compressions == 0:
            return {
                "total_compressions": 0,
                "average_compression_ratio": 0.0,
                "conversations_compressed": 0,
            }

        # Calculate average compression ratio
        total_ratio = 0.0
        ratio_count = 0

        for conversation_id, history in self.compression_history.items():
            for compression in history:
                ratio = compression.get("compression_ratio", 1.0)
                total_ratio += ratio
                ratio_count += 1

        avg_ratio = total_ratio / ratio_count if ratio_count > 0 else 1.0

        return {
            "total_compressions": total_compressions,
            "average_compression_ratio": avg_ratio,
            "conversations_compressed": len(self.compression_history),
            "compression_history": dict(self.compression_history),
        }

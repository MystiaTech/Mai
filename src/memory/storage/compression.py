"""
Progressive conversation compression engine.

This module provides intelligent compression of conversations based on age,
preserving important information while reducing storage requirements.
"""

import re
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from dataclasses import dataclass

try:
    from transformers import pipeline as hf_pipeline

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    hf_pipeline = None

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    nltk = None

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from models.conversation import Message, MessageRole, ConversationMetadata


class CompressionLevel(Enum):
    """Compression levels based on conversation age."""

    FULL = "full"  # 0-7 days: No compression
    KEY_POINTS = "key_points"  # 7-30 days: 70% retention
    SUMMARY = "summary"  # 30-90 days: 40% retention
    METADATA = "metadata"  # 90+ days: Metadata only


@dataclass
class CompressionMetrics:
    """Metrics for compression quality assessment."""

    original_length: int
    compressed_length: int
    compression_ratio: float
    information_retention_score: float
    quality_score: float


@dataclass
class CompressedConversation:
    """Represents a compressed conversation."""

    original_id: str
    compression_level: CompressionLevel
    compressed_at: datetime
    original_created_at: datetime
    content: Union[str, Dict[str, Any]]
    metadata: Dict[str, Any]
    metrics: CompressionMetrics


class CompressionEngine:
    """
    Progressive conversation compression engine.

    Compresses conversations based on age using hybrid extractive-abstractive
    summarization while preserving important information.
    """

    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        """
        Initialize compression engine.

        Args:
            model_name: Name of the summarization model to use
        """
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        self._summarizer = None
        self._initialize_nltk()

    def _initialize_nltk(self) -> None:
        """Initialize NLTK components for extractive summarization."""
        if not NLTK_AVAILABLE:
            self.logger.warning("NLTK not available - using fallback methods")
            return

        try:
            # Download required NLTK data
            import ssl

            try:
                _create_unverified_https_context = ssl._create_unverified_https_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context

            nltk.download("punkt", quiet=True)
            nltk.download("stopwords", quiet=True)
            self.logger.debug("NLTK components initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize NLTK: {e}")

    def _get_summarizer(self):
        """Lazy initialization of summarization pipeline."""
        if TRANSFORMERS_AVAILABLE and self._summarizer is None:
            try:
                self._summarizer = hf_pipeline(
                    "summarization",
                    model=self.model_name,
                    device=-1,  # Use CPU by default
                )
                self.logger.debug(f"Initialized summarizer: {self.model_name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize summarizer: {e}")
                self._summarizer = None
        return self._summarizer

    def get_compression_level(self, age_days: int) -> CompressionLevel:
        """
        Determine compression level based on conversation age.

        Args:
            age_days: Age of conversation in days

        Returns:
            CompressionLevel based on age
        """
        if age_days < 7:
            return CompressionLevel.FULL
        elif age_days < 30:
            return CompressionLevel.KEY_POINTS
        elif age_days < 90:
            return CompressionLevel.SUMMARY
        else:
            return CompressionLevel.METADATA

    def extract_key_points(self, conversation: Dict[str, Any]) -> str:
        """
        Extract key points from conversation using extractive methods.

        Args:
            conversation: Conversation data with messages

        Returns:
            String containing key points
        """
        messages = conversation.get("messages", [])
        if not messages:
            return ""

        # Combine all user and assistant messages
        full_text = ""
        for msg in messages:
            if msg["role"] in ["user", "assistant"]:
                full_text += msg["content"] + "\n"

        if not full_text.strip():
            return ""

        # Extractive summarization using sentence importance
        if not NLTK_AVAILABLE:
            # Simple fallback: split by sentences and take first 70%
            sentences = full_text.split(". ")
            if len(sentences) <= 3:
                return full_text.strip()

            num_sentences = max(3, int(len(sentences) * 0.7))
            key_points = ". ".join(sentences[:num_sentences])
            if not key_points.endswith("."):
                key_points += "."
            return key_points.strip()

        try:
            sentences = sent_tokenize(full_text)
            if len(sentences) <= 3:
                return full_text.strip()

            # Simple scoring based on sentence length and keywords
            scored_sentences = []
            stop_words = set(stopwords.words("english"))

            for i, sentence in enumerate(sentences):
                words = word_tokenize(sentence.lower())
                content_words = [
                    w for w in words if w.isalpha() and w not in stop_words
                ]

                # Score based on length, position, and content word ratio
                length_score = min(len(words) / 20, 1.0)  # Normalize to max 20 words
                position_score = (len(sentences) - i) / len(
                    sentences
                )  # Earlier sentences get higher score
                content_score = len(content_words) / max(len(words), 1)

                total_score = (
                    length_score * 0.3 + position_score * 0.3 + content_score * 0.4
                )
                scored_sentences.append((sentence, total_score))

            # Select top sentences (70% retention)
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            num_sentences = max(3, int(len(sentences) * 0.7))

            key_points = " ".join([s[0] for s in scored_sentences[:num_sentences]])
            return key_points.strip()

        except Exception as e:
            self.logger.error(f"Extractive summarization failed: {e}")
            return full_text[:500] + "..." if len(full_text) > 500 else full_text

    def generate_summary(
        self, conversation: Dict[str, Any], target_ratio: float = 0.4
    ) -> str:
        """
        Generate abstractive summary using transformer model.

        Args:
            conversation: Conversation data with messages
            target_ratio: Target compression ratio (e.g., 0.4 = 40% retention)

        Returns:
            Generated summary string
        """
        messages = conversation.get("messages", [])
        if not messages:
            return ""

        # Combine messages into a single text
        full_text = ""
        for msg in messages:
            if msg["role"] in ["user", "assistant"]:
                full_text += f"{msg['role']}: {msg['content']}\n"

        if not full_text.strip():
            return ""

        # Try abstractive summarization
        summarizer = self._get_summarizer()
        if summarizer:
            try:
                # Calculate target length based on ratio
                max_length = max(50, int(len(full_text.split()) * target_ratio))
                min_length = max(25, int(max_length * 0.5))

                result = summarizer(
                    full_text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False,
                )

                if result and len(result) > 0:
                    summary = result[0].get("summary_text", "")
                    if summary:
                        return summary.strip()

            except Exception as e:
                self.logger.error(f"Abstractive summarization failed: {e}")

        # Fallback to extractive method
        return self.extract_key_points(conversation)

    def extract_metadata_only(self, conversation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract only metadata from conversation.

        Args:
            conversation: Conversation data

        Returns:
            Dictionary with conversation metadata
        """
        messages = conversation.get("messages", [])

        # Extract key metadata
        metadata = {
            "id": conversation.get("id"),
            "title": conversation.get("title"),
            "created_at": conversation.get("created_at"),
            "updated_at": conversation.get("updated_at"),
            "total_messages": len(messages),
            "session_id": conversation.get("session_id"),
            "topics": self._extract_topics(messages),
            "key_entities": self._extract_entities(messages),
            "summary_stats": self._calculate_summary_stats(messages),
        }

        return metadata

    def _extract_topics(self, messages: List[Dict[str, Any]]) -> List[str]:
        """Extract main topics from conversation."""
        topics = set()

        # Simple keyword-based topic extraction
        topic_keywords = {
            "technical": [
                "code",
                "programming",
                "algorithm",
                "function",
                "bug",
                "debug",
            ],
            "personal": ["feel", "think", "opinion", "prefer", "like"],
            "work": ["project", "task", "deadline", "meeting", "team"],
            "learning": ["learn", "study", "understand", "explain", "tutorial"],
            "planning": ["plan", "schedule", "organize", "goal", "strategy"],
        }

        for msg in messages:
            if msg["role"] in ["user", "assistant"]:
                content = msg["content"].lower()
                for topic, keywords in topic_keywords.items():
                    if isinstance(keywords, str):
                        keywords = [keywords]
                    if any(keyword in content for keyword in keywords):
                        topics.add(topic)

        return list(topics)

    def _extract_entities(self, messages: List[Dict[str, Any]]) -> List[str]:
        """Extract key entities from conversation."""
        entities = set()

        # Simple pattern-based entity extraction
        patterns = {
            "emails": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "urls": r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "file_paths": r'\b[a-zA-Z]:\\[^<>:"|?*\n]*\b|\b/[^<>:"|?*\n]*\b',
        }

        for msg in messages:
            if msg["role"] in ["user", "assistant"]:
                content = msg["content"]
                for entity_type, pattern in patterns.items():
                    matches = re.findall(pattern, content)
                    entities.update(matches)

        return list(entities)

    def _calculate_summary_stats(
        self, messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate summary statistics for conversation."""
        user_messages = [m for m in messages if m["role"] == "user"]
        assistant_messages = [m for m in messages if m["role"] == "assistant"]

        total_tokens = sum(m.get("token_count", 0) for m in messages)
        avg_importance = sum(m.get("importance_score", 0.5) for m in messages) / max(
            len(messages), 1
        )

        return {
            "user_message_count": len(user_messages),
            "assistant_message_count": len(assistant_messages),
            "total_tokens": total_tokens,
            "average_importance_score": avg_importance,
            "duration_days": self._calculate_conversation_duration(messages),
        }

    def _calculate_conversation_duration(self, messages: List[Dict[str, Any]]) -> int:
        """Calculate conversation duration in days."""
        if not messages:
            return 0

        timestamps = []
        for msg in messages:
            if "timestamp" in msg:
                try:
                    ts = datetime.fromisoformat(msg["timestamp"])
                    timestamps.append(ts)
                except:
                    continue

        if len(timestamps) < 2:
            return 0

        duration = max(timestamps) - min(timestamps)
        return max(0, duration.days)

    def compress_by_age(self, conversation: Dict[str, Any]) -> CompressedConversation:
        """
        Compress conversation based on its age.

        Args:
            conversation: Conversation data to compress

        Returns:
            CompressedConversation with appropriate compression level
        """
        # Calculate age
        created_at = conversation.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now()

        age_days = (datetime.now() - created_at).days
        compression_level = self.get_compression_level(age_days)

        # Get original content length
        original_content = json.dumps(conversation, ensure_ascii=False)
        original_length = len(original_content)

        # Apply compression based on level
        if compression_level == CompressionLevel.FULL:
            compressed_content = conversation
        elif compression_level == CompressionLevel.KEY_POINTS:
            compressed_content = self.extract_key_points(conversation)
        elif compression_level == CompressionLevel.SUMMARY:
            compressed_content = self.generate_summary(conversation, target_ratio=0.4)
        else:  # METADATA
            compressed_content = self.extract_metadata_only(conversation)

        # Calculate compression metrics
        compressed_content_str = (
            json.dumps(compressed_content, ensure_ascii=False)
            if not isinstance(compressed_content, str)
            else compressed_content
        )
        compressed_length = len(compressed_content_str)
        compression_ratio = compressed_length / max(original_length, 1)

        # Calculate information retention score
        retention_score = self._calculate_retention_score(compression_level)
        quality_score = self._calculate_quality_score(
            compressed_content, conversation, compression_level
        )

        metrics = CompressionMetrics(
            original_length=original_length,
            compressed_length=compressed_length,
            compression_ratio=compression_ratio,
            information_retention_score=retention_score,
            quality_score=quality_score,
        )

        return CompressedConversation(
            original_id=conversation.get("id", "unknown"),
            compression_level=compression_level,
            compressed_at=datetime.now(),
            original_created_at=created_at,
            content=compressed_content,
            metadata={
                "compression_method": "hybrid_extractive_abstractive",
                "age_days": age_days,
                "original_tokens": conversation.get("total_tokens", 0),
            },
            metrics=metrics,
        )

    def _calculate_retention_score(self, compression_level: CompressionLevel) -> float:
        """Calculate information retention score based on compression level."""
        retention_map = {
            CompressionLevel.FULL: 1.0,
            CompressionLevel.KEY_POINTS: 0.7,
            CompressionLevel.SUMMARY: 0.4,
            CompressionLevel.METADATA: 0.1,
        }
        return retention_map.get(compression_level, 0.1)

    def _calculate_quality_score(
        self,
        compressed_content: Union[str, Dict[str, Any]],
        original: Dict[str, Any],
        level: CompressionLevel,
    ) -> float:
        """
        Calculate quality score for compressed content.

        Args:
            compressed_content: The compressed content
            original: Original conversation
            level: Compression level used

        Returns:
            Quality score between 0.0 and 1.0
        """
        try:
            # Base score from compression level
            base_scores = {
                CompressionLevel.FULL: 1.0,
                CompressionLevel.KEY_POINTS: 0.8,
                CompressionLevel.SUMMARY: 0.7,
                CompressionLevel.METADATA: 0.5,
            }
            base_score = base_scores.get(level, 0.5)

            # Adjust based on content quality
            if isinstance(compressed_content, str):
                # Check for common quality indicators
                content_length = len(compressed_content)
                if content_length == 0:
                    return 0.0

                # Penalize very short content
                if level in [CompressionLevel.KEY_POINTS, CompressionLevel.SUMMARY]:
                    if content_length < 50:
                        base_score *= 0.5
                    elif content_length < 100:
                        base_score *= 0.8

                # Check for coherent structure
                sentences = (
                    compressed_content.count(".")
                    + compressed_content.count("!")
                    + compressed_content.count("?")
                )
                if sentences > 0:
                    coherence_score = min(
                        sentences / 10, 1.0
                    )  # More sentences = more coherent
                    base_score = (base_score + coherence_score) / 2

            return max(0.0, min(1.0, base_score))

        except Exception as e:
            self.logger.error(f"Error calculating quality score: {e}")
            return 0.5

    def decompress(self, compressed: CompressedConversation) -> Dict[str, Any]:
        """
        Decompress compressed conversation to summary view.

        Args:
            compressed: Compressed conversation to decompress

        Returns:
            Summary view of the conversation
        """
        if compressed.compression_level == CompressionLevel.FULL:
            # Return full conversation if no compression
            return (
                compressed.content
                if isinstance(compressed.content, dict)
                else {"summary": compressed.content}
            )

        # Create summary view for compressed conversations
        summary = {
            "id": compressed.original_id,
            "compression_level": compressed.compression_level.value,
            "compressed_at": compressed.compressed_at.isoformat(),
            "original_created_at": compressed.original_created_at.isoformat(),
            "metadata": compressed.metadata,
            "metrics": {
                "compression_ratio": compressed.metrics.compression_ratio,
                "information_retention_score": compressed.metrics.information_retention_score,
                "quality_score": compressed.metrics.quality_score,
            },
        }

        if compressed.compression_level == CompressionLevel.METADATA:
            # Content is already metadata
            if isinstance(compressed.content, dict):
                summary["metadata"].update(compressed.content)
            summary["summary"] = "Metadata only - full content compressed due to age"
        else:
            # Content is key points or summary text
            summary["summary"] = compressed.content

        return summary

    def batch_compress_conversations(
        self, conversations: List[Dict[str, Any]]
    ) -> List[CompressedConversation]:
        """
        Compress multiple conversations efficiently.

        Args:
            conversations: List of conversations to compress

        Returns:
            List of compressed conversations
        """
        compressed_list = []

        for conversation in conversations:
            try:
                compressed = self.compress_by_age(conversation)
                compressed_list.append(compressed)
            except Exception as e:
                self.logger.error(
                    f"Failed to compress conversation {conversation.get('id', 'unknown')}: {e}"
                )
                continue

        self.logger.info(
            f"Compressed {len(compressed_list)}/{len(conversations)} conversations successfully"
        )
        return compressed_list

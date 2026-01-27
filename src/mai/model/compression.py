"""
Context compression and token management for Mai.

Handles conversation context within model token limits while preserving
important information and conversation quality.
"""

import re
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import deque
import hashlib
import json
import time


@dataclass
class TokenInfo:
    """Token counting information."""

    count: int
    model_name: str
    accuracy: float = 0.95  # Confidence in token count accuracy


@dataclass
class CompressionResult:
    """Result of context compression."""

    compressed_conversation: List[Dict[str, Any]]
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    quality_score: float
    preserved_elements: List[str]


@dataclass
class BudgetEnforcement:
    """Token budget enforcement result."""

    action: str  # 'proceed', 'compress', 'reject'
    token_count: int
    budget_limit: int
    urgency: float  # 0.0 to 1.0
    message: str


class ContextCompressor:
    """
    Handles context compression and token management for conversations.

    Features:
    - Token counting with model-specific accuracy
    - Intelligent compression preserving key information
    - Budget enforcement to prevent exceeding context windows
    - Quality metrics and validation
    """

    def __init__(self):
        """Initialize the context compressor."""
        self.tiktoken_available = self._check_tiktoken()
        if self.tiktoken_available:
            import tiktoken

            self.encoders = {
                "gpt-3.5-turbo": tiktoken.encoding_for_model("gpt-3.5-turbo"),
                "gpt-4": tiktoken.encoding_for_model("gpt-4"),
                "gpt-4-turbo": tiktoken.encoding_for_model("gpt-4-turbo"),
                "text-davinci-003": tiktoken.encoding_for_model("text-davinci-003"),
            }
        else:
            self.encoders = {}
            print("Warning: tiktoken not available, using approximate token counting")

        # Compression thresholds
        self.warning_threshold = 0.75  # Warn at 75% of context window
        self.critical_threshold = 0.90  # Critical at 90% of context window
        self.budget_ratio = 0.9  # Budget at 90% of context window

        # Compression cache
        self.compression_cache = {}
        self.cache_ttl = 3600  # 1 hour
        self.performance_cache = deque(maxlen=100)

        # Quality metrics
        self.min_quality_score = 0.7
        self.preservation_patterns = [
            r"\b(install|configure|set up|create|build|implement)\b",
            r"\b(error|bug|issue|problem|fix)\b",
            r"\b(decision|choice|prefer|selected)\b",
            r"\b(important|critical|essential|must)\b",
            r"\b(key|main|primary|core)\b",
        ]

    def _check_tiktoken(self) -> bool:
        """Check if tiktoken is available."""
        try:
            import tiktoken

            return True
        except ImportError:
            return False

    def count_tokens(self, text: str, model_name: str = "gpt-3.5-turbo") -> TokenInfo:
        """
        Count tokens in text with model-specific accuracy.

        Args:
            text: Text to count tokens for
            model_name: Model name for tokenization

        Returns:
            TokenInfo with count and accuracy
        """
        if not text:
            return TokenInfo(0, model_name, 1.0)

        if self.tiktoken_available and model_name in self.encoders:
            encoder = self.encoders[model_name]
            try:
                tokens = encoder.encode(text)
                return TokenInfo(len(tokens), model_name, 0.99)
            except Exception as e:
                print(f"Tiktoken error: {e}, falling back to approximation")

        # Fallback: approximate token counting
        # Rough approximation: ~4 characters per token for English
        # Slightly better approach using word and punctuation patterns
        words = re.findall(r"\w+|[^\w\s]", text)
        # Adjust for model families
        model_multipliers = {
            "gpt-3.5": 1.0,
            "gpt-4": 0.9,  # More efficient tokenization
            "claude": 1.1,  # Less efficient
            "llama": 1.2,  # Even less efficient
        }

        # Determine model family
        model_family = "gpt-3.5"
        for family in model_multipliers:
            if family in model_name.lower():
                model_family = family
                break

        multiplier = model_multipliers.get(model_family, 1.0)
        token_count = int(len(words) * 1.3 * multiplier)  # 1.3 is base conversion

        return TokenInfo(token_count, model_name, 0.85)  # Lower accuracy for approximation

    def should_compress(
        self, conversation: List[Dict[str, Any]], model_context_window: int
    ) -> Tuple[bool, float, str]:
        """
        Determine if conversation should be compressed.

        Args:
            conversation: List of message dictionaries
            model_context_window: Model's context window size

        Returns:
            Tuple of (should_compress, urgency, message)
        """
        total_tokens = sum(self.count_tokens(msg.get("content", "")).count for msg in conversation)

        usage_ratio = total_tokens / model_context_window

        if usage_ratio >= self.critical_threshold:
            return True, 1.0, f"Critical: {usage_ratio:.1%} of context window used"
        elif usage_ratio >= self.warning_threshold:
            return True, 0.7, f"Warning: {usage_ratio:.1%} of context window used"
        elif len(conversation) > 50:  # Conversation length consideration
            return True, 0.5, "Long conversation: consider compression for performance"
        else:
            return False, 0.0, "Context within acceptable limits"

    def preserve_key_elements(self, conversation: List[Dict[str, Any]]) -> List[str]:
        """
        Extract and preserve critical information from conversation.

        Args:
            conversation: List of message dictionaries

        Returns:
            List of critical elements to preserve
        """
        key_elements = []

        for msg in conversation:
            content = msg.get("content", "")
            role = msg.get("role", "")

            # Look for important patterns
            for pattern in self.preservation_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    # Extract surrounding context
                    for match in matches:
                        # Find the sentence containing the match
                        sentences = re.split(r"[.!?]+", content)
                        for sentence in sentences:
                            if match.lower() in sentence.lower():
                                key_elements.append(f"{role}: {sentence.strip()}")
                                break

        # Also preserve system messages and instructions
        for msg in conversation:
            if msg.get("role") in ["system", "instruction"]:
                key_elements.append(f"system: {msg.get('content', '')}")

        return key_elements

    def compress_conversation(
        self, conversation: List[Dict[str, Any]], target_token_ratio: float = 0.5
    ) -> CompressionResult:
        """
        Compress conversation while preserving key information.

        Args:
            conversation: List of message dictionaries
            target_token_ratio: Target ratio of original tokens to keep

        Returns:
            CompressionResult with compressed conversation and metrics
        """
        if not conversation:
            return CompressionResult([], 0, 0, 1.0, 1.0, [])

        # Calculate current token usage
        original_tokens = sum(
            self.count_tokens(msg.get("content", "")).count for msg in conversation
        )

        target_tokens = int(original_tokens * target_token_ratio)

        # Check cache
        cache_key = self._get_cache_key(conversation, target_token_ratio)
        if cache_key in self.compression_cache:
            cached_result = self.compression_cache[cache_key]
            if time.time() - cached_result["timestamp"] < self.cache_ttl:
                return CompressionResult(**cached_result["result"])

        # Preserve key elements
        key_elements = self.preserve_key_elements(conversation)

        # Split conversation: keep recent messages, compress older ones
        split_point = max(0, len(conversation) // 2)  # Keep second half
        recent_messages = conversation[split_point:]
        older_messages = conversation[:split_point]

        compressed_messages = []

        # Summarize older messages
        if older_messages:
            summary = self._create_summary(older_messages, target_tokens // 2)
            compressed_messages.append(
                {
                    "role": "system",
                    "content": f"[Compressed context: {summary}]",
                    "metadata": {
                        "compressed": True,
                        "original_count": len(older_messages),
                        "summary_token_count": self.count_tokens(summary).count,
                    },
                }
            )

        # Add recent messages
        compressed_messages.extend(recent_messages)

        # Add key elements if they might be lost
        if key_elements:
            key_content = "\n\nKey information to remember:\n" + "\n".join(key_elements[:5])
            compressed_messages.append(
                {
                    "role": "system",
                    "content": key_content,
                    "metadata": {"type": "key_elements", "preserved_count": len(key_elements)},
                }
            )

        # Calculate metrics
        compressed_tokens = sum(
            self.count_tokens(msg.get("content", "")).count for msg in compressed_messages
        )

        compression_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0
        quality_score = self._calculate_quality_score(
            conversation, compressed_messages, key_elements
        )

        result = CompressionResult(
            compressed_conversation=compressed_messages,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compression_ratio,
            quality_score=quality_score,
            preserved_elements=key_elements,
        )

        # Cache result
        self.compression_cache[cache_key] = {"result": result.__dict__, "timestamp": time.time()}

        return result

    def _create_summary(self, messages: List[Dict[str, Any]], target_tokens: int) -> str:
        """
        Create a summary of older messages.

        Args:
            messages: List of message dictionaries
            target_tokens: Target token count for summary

        Returns:
            Summary string
        """
        # Extract key points from messages
        key_points = []

        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "")

            # Extract first sentence or important parts
            sentences = re.split(r"[.!?]+", content)
            if sentences:
                first_sentence = sentences[0].strip()
                if len(first_sentence) > 10:  # Skip very short fragments
                    key_points.append(f"{role}: {first_sentence}")

        # Join and truncate to target length
        summary = " | ".join(key_points)

        # Truncate if too long
        while len(summary) > target_tokens * 4 and key_points:  # Rough character estimate
            key_points.pop()
            summary = " | ".join(key_points)

        return summary if summary else "Previous conversation context"

    def _calculate_quality_score(
        self,
        original: List[Dict[str, Any]],
        compressed: List[Dict[str, Any]],
        preserved_elements: List[str],
    ) -> float:
        """
        Calculate quality score for compression.

        Args:
            original: Original conversation
            compressed: Compressed conversation
            preserved_elements: Elements preserved

        Returns:
            Quality score between 0.0 and 1.0
        """
        # Base score from token preservation
        original_tokens = sum(self.count_tokens(msg.get("content", "")).count for msg in original)
        compressed_tokens = sum(
            self.count_tokens(msg.get("content", "")).count for msg in compressed
        )

        preservation_score = min(1.0, compressed_tokens / original_tokens)

        # Bonus for preserved elements
        element_bonus = min(0.2, len(preserved_elements) * 0.02)

        # Penalty for too aggressive compression
        if compressed_tokens < original_tokens * 0.3:
            preservation_score *= 0.8

        quality_score = min(1.0, preservation_score + element_bonus)

        return quality_score

    def enforce_token_budget(
        self,
        conversation: List[Dict[str, Any]],
        model_context_window: int,
        budget_ratio: Optional[float] = None,
    ) -> BudgetEnforcement:
        """
        Enforce token budget before model call.

        Args:
            conversation: List of message dictionaries
            model_context_window: Model's context window size
            budget_ratio: Budget ratio (default from config)

        Returns:
            BudgetEnforcement with action and details
        """
        if budget_ratio is None:
            budget_ratio = self.budget_ratio

        budget_limit = int(model_context_window * budget_ratio)
        current_tokens = sum(
            self.count_tokens(msg.get("content", "")).count for msg in conversation
        )

        usage_ratio = current_tokens / model_context_window

        if current_tokens > budget_limit:
            if usage_ratio >= 0.95:
                return BudgetEnforcement(
                    action="reject",
                    token_count=current_tokens,
                    budget_limit=budget_limit,
                    urgency=1.0,
                    message=f"Conversation too long: {current_tokens} tokens exceeds budget of {budget_limit}",
                )
            else:
                return BudgetEnforcement(
                    action="compress",
                    token_count=current_tokens,
                    budget_limit=budget_limit,
                    urgency=min(1.0, usage_ratio),
                    message=f"Compression needed: {current_tokens} tokens exceeds budget of {budget_limit}",
                )
        else:
            urgency = max(0.0, usage_ratio - 0.7) / 0.2  # Normalize between 0.7-0.9
            return BudgetEnforcement(
                action="proceed",
                token_count=current_tokens,
                budget_limit=budget_limit,
                urgency=urgency,
                message=f"Within budget: {current_tokens} tokens of {budget_limit}",
            )

    def validate_compression(
        self, original: List[Dict[str, Any]], compressed: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate compression quality and information preservation.

        Args:
            original: Original conversation
            compressed: Compressed conversation

        Returns:
            Dictionary with validation metrics
        """
        # Token-based metrics
        original_tokens = sum(self.count_tokens(msg.get("content", "")).count for msg in original)
        compressed_tokens = sum(
            self.count_tokens(msg.get("content", "")).count for msg in compressed
        )

        # Semantic similarity (simplified)
        original_text = " ".join(msg.get("content", "") for msg in original).lower()
        compressed_text = " ".join(msg.get("content", "") for msg in compressed).lower()

        # Word overlap as simple similarity metric
        original_words = set(re.findall(r"\w+", original_text))
        compressed_words = set(re.findall(r"\w+", compressed_text))

        if original_words:
            similarity = len(original_words & compressed_words) / len(original_words)
        else:
            similarity = 1.0

        # Key information preservation
        original_key = self.preserve_key_elements(original)
        compressed_key = self.preserve_key_elements(compressed)

        key_preservation = len(compressed_key) / max(1, len(original_key))

        return {
            "token_preservation": compressed_tokens / max(1, original_tokens),
            "semantic_similarity": similarity,
            "key_information_preservation": key_preservation,
            "overall_quality": (similarity + key_preservation) / 2,
            "recommendations": self._get_validation_recommendations(
                similarity, key_preservation, compressed_tokens / max(1, original_tokens)
            ),
        }

    def _get_validation_recommendations(
        self, similarity: float, key_preservation: float, token_ratio: float
    ) -> List[str]:
        """Get recommendations based on validation metrics."""
        recommendations = []

        if similarity < 0.7:
            recommendations.append("Low semantic similarity - consider preserving more context")

        if key_preservation < 0.8:
            recommendations.append(
                "Key information not well preserved - adjust preservation patterns"
            )

        if token_ratio > 0.8:
            recommendations.append("Compression too conservative - can reduce more")
        elif token_ratio < 0.3:
            recommendations.append("Compression too aggressive - losing too much content")

        if not recommendations:
            recommendations.append("Compression quality is acceptable")

        return recommendations

    def _get_cache_key(self, conversation: List[Dict[str, Any]], target_ratio: float) -> str:
        """Generate cache key for compression result."""
        # Create hash of conversation and target ratio
        content = json.dumps([msg.get("content", "") for msg in conversation], sort_keys=True)
        content_hash = hashlib.md5(content.encode()).hexdigest()
        return f"{content_hash}_{target_ratio}"

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the compressor."""
        return {
            "cache_size": len(self.compression_cache),
            "cache_hit_ratio": len(self.performance_cache) / max(1, len(self.compression_cache)),
            "tiktoken_available": self.tiktoken_available,
            "supported_models": list(self.encoders.keys()) if self.tiktoken_available else [],
            "compression_thresholds": {
                "warning": self.warning_threshold,
                "critical": self.critical_threshold,
                "budget": self.budget_ratio,
            },
        }

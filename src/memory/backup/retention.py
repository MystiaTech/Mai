"""
Smart retention policies for conversation preservation.

Implements value-based retention scoring that keeps important
conversations longer while efficiently managing storage usage.
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
import statistics

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from memory.storage.sqlite_manager import SQLiteManager


class RetentionPolicy:
    """
    Smart retention policy engine.

    Calculates conversation importance scores and determines
    which conversations should be retained or compressed.
    """

    def __init__(self, sqlite_manager: SQLiteManager):
        """
        Initialize retention policy.

        Args:
            sqlite_manager: SQLite manager instance for data access
        """
        self.db_manager = sqlite_manager
        self.logger = logging.getLogger(__name__)

        # Retention policy parameters
        self.important_threshold = 0.7  # Above this = retain full
        self.preserve_threshold = 0.4  # Above this = lighter compression
        self.user_marked_multiplier = 1.5  # Boost for user-marked important

        # Engagement scoring weights
        self.weights = {
            "message_count": 0.2,  # More messages = higher engagement
            "response_quality": 0.25,  # Back-and-forth conversation
            "topic_diversity": 0.15,  # Multiple topics = important
            "time_span": 0.1,  # Longer duration = important
            "user_marked": 0.2,  # User explicitly marked important
            "question_density": 0.1,  # Questions = seeking information
        }

    def calculate_importance_score(self, conversation: Dict[str, Any]) -> float:
        """
        Calculate importance score for a conversation.

        Args:
            conversation: Conversation data with messages and metadata

        Returns:
            Importance score between 0.0 and 1.0
        """
        try:
            messages = conversation.get("messages", [])
            if not messages:
                return 0.0

            # Extract basic metrics
            message_count = len(messages)
            user_messages = [m for m in messages if m["role"] == "user"]
            assistant_messages = [m for m in messages if m["role"] == "assistant"]

            # Calculate engagement metrics
            scores = {}

            # 1. Message count score (normalized)
            scores["message_count"] = min(
                message_count / 20, 1.0
            )  # 20 messages = full score

            # 2. Response quality (back-and-forth ratio)
            if len(user_messages) > 0 and len(assistant_messages) > 0:
                ratio = min(len(assistant_messages), len(user_messages)) / max(
                    len(assistant_messages), len(user_messages)
                )
                scores["response_quality"] = ratio  # Close to 1.0 = good conversation
            else:
                scores["response_quality"] = 0.5

            # 3. Topic diversity (variety in content)
            scores["topic_diversity"] = self._calculate_topic_diversity(messages)

            # 4. Time span (conversation duration)
            scores["time_span"] = self._calculate_time_span_score(messages)

            # 5. User marked important
            metadata = conversation.get("metadata", {})
            user_marked = metadata.get("user_marked_important", False)
            scores["user_marked"] = self.user_marked_multiplier if user_marked else 1.0

            # 6. Question density (information seeking)
            scores["question_density"] = self._calculate_question_density(user_messages)

            # Calculate weighted final score
            final_score = 0.0
            for factor, weight in self.weights.items():
                final_score += scores.get(factor, 0.0) * weight

            # Normalize to 0-1 range
            final_score = max(0.0, min(1.0, final_score))

            self.logger.debug(
                f"Importance score for {conversation.get('id')}: {final_score:.3f}"
            )
            return final_score

        except Exception as e:
            self.logger.error(f"Failed to calculate importance score: {e}")
            return 0.5  # Default to neutral

    def _calculate_topic_diversity(self, messages: List[Dict[str, Any]]) -> float:
        """Calculate topic diversity score from messages."""
        try:
            # Simple topic-based diversity using keyword categories
            topic_keywords = {
                "technical": [
                    "code",
                    "programming",
                    "algorithm",
                    "function",
                    "bug",
                    "debug",
                    "api",
                    "database",
                ],
                "personal": [
                    "feel",
                    "think",
                    "opinion",
                    "prefer",
                    "like",
                    "personal",
                    "life",
                ],
                "work": [
                    "project",
                    "task",
                    "deadline",
                    "meeting",
                    "team",
                    "work",
                    "job",
                ],
                "learning": [
                    "learn",
                    "study",
                    "understand",
                    "explain",
                    "tutorial",
                    "help",
                ],
                "planning": ["plan", "schedule", "organize", "goal", "strategy"],
                "creative": ["design", "create", "write", "art", "music", "story"],
            }

            topic_counts = defaultdict(int)
            total_content = ""

            for message in messages:
                if message["role"] in ["user", "assistant"]:
                    content = message["content"].lower()
                    total_content += content + " "

                    # Count topic occurrences
                    for topic, keywords in topic_keywords.items():
                        for keyword in keywords:
                            if keyword in content:
                                topic_counts[topic] += 1

            # Diversity = number of topics with significant presence
            significant_topics = sum(1 for count in topic_counts.values() if count >= 2)
            diversity_score = min(significant_topics / len(topic_keywords), 1.0)

            return diversity_score

        except Exception as e:
            self.logger.error(f"Failed to calculate topic diversity: {e}")
            return 0.5

    def _calculate_time_span_score(self, messages: List[Dict[str, Any]]) -> float:
        """Calculate time span score based on conversation duration."""
        try:
            timestamps = []
            for message in messages:
                if "timestamp" in message:
                    try:
                        ts = datetime.fromisoformat(message["timestamp"])
                        timestamps.append(ts)
                    except:
                        continue

            if len(timestamps) < 2:
                return 0.1  # Very short conversation

            duration = max(timestamps) - min(timestamps)
            duration_hours = duration.total_seconds() / 3600

            # Score based on duration (24 hours = full score)
            return min(duration_hours / 24, 1.0)

        except Exception as e:
            self.logger.error(f"Failed to calculate time span: {e}")
            return 0.5

    def _calculate_question_density(self, user_messages: List[Dict[str, Any]]) -> float:
        """Calculate question density from user messages."""
        try:
            if not user_messages:
                return 0.0

            question_count = 0
            total_words = 0

            for message in user_messages:
                content = message["content"]
                # Count questions
                question_marks = content.count("?")
                question_words = len(
                    re.findall(
                        r"\b(how|what|when|where|why|which|who|can|could|would|should|is|are|do|does)\b",
                        content,
                        re.IGNORECASE,
                    )
                )
                question_count += question_marks + question_words

                # Count words
                words = len(content.split())
                total_words += words

            if total_words == 0:
                return 0.0

            question_ratio = question_count / total_words
            return min(question_ratio * 5, 1.0)  # Normalize

        except Exception as e:
            self.logger.error(f"Failed to calculate question density: {e}")
            return 0.5

    def should_retain_full(
        self, conversation: Dict[str, Any], importance_score: Optional[float] = None
    ) -> bool:
        """
        Determine if conversation should be retained in full form.

        Args:
            conversation: Conversation data
            importance_score: Pre-calculated importance score (optional)

        Returns:
            True if conversation should be retained full
        """
        if importance_score is None:
            importance_score = self.calculate_importance_score(conversation)

        # User explicitly marked important always retained
        metadata = conversation.get("metadata", {})
        if metadata.get("user_marked_important", False):
            return True

        # High importance score
        if importance_score >= self.important_threshold:
            return True

        # Recent important conversations (within 30 days)
        created_at = conversation.get("created_at")
        if created_at:
            try:
                conv_date = datetime.fromisoformat(created_at)
                if (datetime.now() - conv_date).days <= 30 and importance_score >= 0.5:
                    return True
            except:
                pass

        return False

    def should_retain_compressed(
        self, conversation: Dict[str, Any], importance_score: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        Determine if conversation should be compressed and to what level.

        Args:
            conversation: Conversation data
            importance_score: Pre-calculated importance score (optional)

        Returns:
            Tuple of (should_compress, recommended_compression_level)
        """
        if importance_score is None:
            importance_score = self.calculate_importance_score(conversation)

        # Check if should retain full
        if self.should_retain_full(conversation, importance_score):
            return False, "full"

        # Determine compression level based on importance
        if importance_score >= self.preserve_threshold:
            # Important: lighter compression (key points)
            return True, "key_points"
        elif importance_score >= 0.2:
            # Moderately important: summary compression
            return True, "summary"
        else:
            # Low importance: metadata only
            return True, "metadata"

    def update_retention_policy(self, policy_settings: Dict[str, Any]) -> None:
        """
        Update retention policy parameters.

        Args:
            policy_settings: Dictionary of policy parameter updates
        """
        try:
            if "important_threshold" in policy_settings:
                self.important_threshold = float(policy_settings["important_threshold"])
            if "preserve_threshold" in policy_settings:
                self.preserve_threshold = float(policy_settings["preserve_threshold"])
            if "user_marked_multiplier" in policy_settings:
                self.user_marked_multiplier = float(
                    policy_settings["user_marked_multiplier"]
                )
            if "weights" in policy_settings:
                self.weights.update(policy_settings["weights"])

            self.logger.info(f"Updated retention policy: {policy_settings}")

        except Exception as e:
            self.logger.error(f"Failed to update retention policy: {e}")

    def get_retention_recommendations(
        self, conversations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Get retention recommendations for multiple conversations.

        Args:
            conversations: List of conversations to analyze

        Returns:
            List of recommendations with scores and actions
        """
        recommendations = []

        for conversation in conversations:
            try:
                importance_score = self.calculate_importance_score(conversation)
                should_compress, compression_level = self.should_retain_compressed(
                    conversation, importance_score
                )

                recommendation = {
                    "conversation_id": conversation.get("id"),
                    "title": conversation.get("title"),
                    "created_at": conversation.get("created_at"),
                    "importance_score": importance_score,
                    "should_compress": should_compress,
                    "recommended_level": compression_level,
                    "user_marked_important": conversation.get("metadata", {}).get(
                        "user_marked_important", False
                    ),
                    "message_count": len(conversation.get("messages", [])),
                    "retention_reason": self._get_retention_reason(
                        importance_score, compression_level
                    ),
                }

                recommendations.append(recommendation)

            except Exception as e:
                self.logger.error(
                    f"Failed to analyze conversation {conversation.get('id')}: {e}"
                )
                continue

        # Sort by importance score (highest first)
        recommendations.sort(key=lambda x: x["importance_score"], reverse=True)
        return recommendations

    def _get_retention_reason(
        self, importance_score: float, compression_level: str
    ) -> str:
        """Get human-readable reason for retention decision."""
        if compression_level == "full":
            if importance_score >= self.important_threshold:
                return "High importance - retained full"
            else:
                return "Recent conversation - retained full"
        elif compression_level == "key_points":
            return f"Moderate importance ({importance_score:.2f}) - key points retained"
        elif compression_level == "summary":
            return f"Standard importance ({importance_score:.2f}) - summary compression"
        else:
            return f"Low importance ({importance_score:.2f}) - metadata only"

    def mark_conversation_important(
        self, conversation_id: str, important: bool = True
    ) -> bool:
        """
        Mark a conversation as user-important.

        Args:
            conversation_id: ID of conversation to mark
            important: Whether to mark as important (True) or not important (False)

        Returns:
            True if marked successfully
        """
        try:
            conversation = self.db_manager.get_conversation(
                conversation_id, include_messages=False
            )
            if not conversation:
                self.logger.error(f"Conversation {conversation_id} not found")
                return False

            # Update metadata
            metadata = conversation.get("metadata", {})
            metadata["user_marked_important"] = important
            metadata["marked_important_at"] = datetime.now().isoformat()

            self.db_manager.update_conversation_metadata(conversation_id, metadata)

            self.logger.info(
                f"Marked conversation {conversation_id} as {'important' if important else 'not important'}"
            )
            return True

        except Exception as e:
            self.logger.error(
                f"Failed to mark conversation {conversation_id} important: {e}"
            )
            return False

    def get_important_conversations(self) -> List[Dict[str, Any]]:
        """
        Get all user-marked important conversations.

        Returns:
            List of important conversations
        """
        try:
            recent_conversations = self.db_manager.get_recent_conversations(limit=1000)

            important_conversations = []
            for conversation in recent_conversations:
                full_conversation = self.db_manager.get_conversation(
                    conversation["id"], include_messages=True
                )
                if full_conversation:
                    metadata = full_conversation.get("metadata", {})
                    if metadata.get("user_marked_important", False):
                        important_conversations.append(full_conversation)

            return important_conversations

        except Exception as e:
            self.logger.error(f"Failed to get important conversations: {e}")
            return []

    def get_retention_stats(self) -> Dict[str, Any]:
        """
        Get retention policy statistics.

        Returns:
            Dictionary with retention statistics
        """
        try:
            recent_conversations = self.db_manager.get_recent_conversations(limit=500)

            stats = {
                "total_conversations": len(recent_conversations),
                "important_marked": 0,
                "importance_distribution": {"high": 0, "medium": 0, "low": 0},
                "average_importance": 0.0,
                "compression_recommendations": {
                    "full": 0,
                    "key_points": 0,
                    "summary": 0,
                    "metadata": 0,
                },
            }

            importance_scores = []

            for conv_data in recent_conversations:
                conversation = self.db_manager.get_conversation(
                    conv_data["id"], include_messages=True
                )
                if not conversation:
                    continue

                importance_score = self.calculate_importance_score(conversation)
                importance_scores.append(importance_score)

                # Check if user marked important
                metadata = conversation.get("metadata", {})
                if metadata.get("user_marked_important", False):
                    stats["important_marked"] += 1

                # Categorize importance
                if importance_score >= self.important_threshold:
                    stats["importance_distribution"]["high"] += 1
                elif importance_score >= self.preserve_threshold:
                    stats["importance_distribution"]["medium"] += 1
                else:
                    stats["importance_distribution"]["low"] += 1

                # Compression recommendations
                should_compress, level = self.should_retain_compressed(
                    conversation, importance_score
                )
                if level in stats["compression_recommendations"]:
                    stats["compression_recommendations"][level] += 1
                else:
                    stats["compression_recommendations"]["full"] += 1

            if importance_scores:
                stats["average_importance"] = statistics.mean(importance_scores)

            return stats

        except Exception as e:
            self.logger.error(f"Failed to get retention stats: {e}")
            return {}

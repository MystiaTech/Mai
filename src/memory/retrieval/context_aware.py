"""
Context-aware search with topic-based prioritization.

This module provides context-aware search capabilities that prioritize
search results based on current conversation topic and context.
"""

import sys
import os
from typing import List, Optional, Dict, Any, Set
from datetime import datetime
import re
import logging

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from .search_types import SearchResult, SearchQuery


class ContextAwareSearch:
    """
    Context-aware search with topic-based result prioritization.

    Provides intelligent search that considers current conversation context
    and topic relevance when ranking search results.
    """

    def __init__(self, sqlite_manager):
        """
        Initialize context-aware search with SQLite manager.

        Args:
            sqlite_manager: SQLiteManager instance for metadata access
        """
        self.sqlite_manager = sqlite_manager
        self.logger = logging.getLogger(__name__)

        # Simple topic keywords for classification
        self.topic_keywords = {
            "technical": [
                "code",
                "programming",
                "algorithm",
                "function",
                "class",
                "method",
                "api",
                "database",
                "debug",
                "error",
                "test",
                "implementation",
            ],
            "personal": [
                "i",
                "me",
                "my",
                "feel",
                "think",
                "believe",
                "want",
                "need",
                "help",
                "opinion",
                "experience",
            ],
            "question": [
                "what",
                "how",
                "why",
                "when",
                "where",
                "which",
                "can",
                "could",
                "should",
                "would",
                "question",
                "answer",
            ],
            "task": [
                "create",
                "implement",
                "build",
                "develop",
                "design",
                "feature",
                "fix",
                "update",
                "add",
                "remove",
                "modify",
            ],
            "system": [
                "system",
                "performance",
                "resource",
                "memory",
                "storage",
                "optimization",
                "efficiency",
                "architecture",
            ],
        }

    def _extract_keywords(self, text: str) -> Set[str]:
        """
        Extract keywords from text for topic analysis.

        Args:
            text: Text to analyze

        Returns:
            Set of extracted keywords
        """
        # Normalize text
        text = text.lower()

        # Extract words (3+ characters)
        words = set()
        for word in re.findall(r"\b[a-z]{3,}\b", text):
            words.add(word)

        return words

    def _classify_topic(self, text: str) -> str:
        """
        Classify text into topic categories.

        Args:
            text: Text to classify

        Returns:
            Topic classification string
        """
        keywords = self._extract_keywords(text)

        # Score topics based on keyword matches
        topic_scores = {}
        for topic, topic_keywords in self.topic_keywords.items():
            score = sum(1 for keyword in keywords if keyword in topic_keywords)
            if score > 0:
                topic_scores[topic] = score

        if not topic_scores:
            return "general"

        # Return highest scoring topic
        return max(topic_scores.items(), key=lambda x: x[1])[0]

    def _get_current_context(
        self, conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get current conversation context for topic analysis.

        Args:
            conversation_id: Current conversation ID (optional)

        Returns:
            Dictionary with context information
        """
        context = {
            "current_topic": "general",
            "recent_messages": [],
            "active_keywords": set(),
        }

        if conversation_id:
            try:
                # Get recent messages from current conversation
                recent_messages = self.sqlite_manager.get_recent_messages(
                    conversation_id, limit=10
                )

                if recent_messages:
                    context["recent_messages"] = recent_messages

                    # Extract keywords from recent messages
                    all_text = " ".join(
                        [msg.get("content", "") for msg in recent_messages]
                    )
                    context["active_keywords"] = self._extract_keywords(all_text)

                    # Classify current topic
                    context["current_topic"] = self._classify_topic(all_text)

            except Exception as e:
                self.logger.error(f"Failed to get context: {e}")

        return context

    def _calculate_topic_relevance(
        self, result: SearchResult, current_topic: str, active_keywords: Set[str]
    ) -> float:
        """
        Calculate topic relevance score for a search result.

        Args:
            result: SearchResult to score
            current_topic: Current conversation topic
            active_keywords: Keywords active in current conversation

        Returns:
            Topic relevance boost factor (1.0 = no boost, >1.0 = boosted)
        """
        result_keywords = self._extract_keywords(result.content)

        # Topic-based boost
        result_topic = self._classify_topic(result.content)
        topic_boost = 1.0

        if result_topic == current_topic:
            topic_boost = 1.5  # 50% boost for same topic
        elif result_topic in ["technical", "system"] and current_topic in [
            "technical",
            "system",
        ]:
            topic_boost = 1.3  # 30% boost for technical topics

        # Keyword overlap boost
        keyword_overlap = len(result_keywords & active_keywords)
        total_keywords = len(result_keywords) or 1
        keyword_boost = 1.0 + (keyword_overlap / total_keywords) * 0.3  # Max 30% boost

        # Combined boost (limited to prevent over-boosting)
        combined_boost = min(2.0, topic_boost * keyword_boost)

        return float(combined_boost)

    def prioritize_by_topic(
        self,
        results: List[SearchResult],
        current_topic: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Prioritize search results based on current conversation topic.

        Args:
            results: List of search results to prioritize
            current_topic: Current topic (auto-detected if None)
            conversation_id: Current conversation ID (for context analysis)

        Returns:
            Reordered list of search results with topic-based scoring
        """
        if not results:
            return []

        # Get current context
        context = self._get_current_context(conversation_id)

        # Use provided topic or auto-detect
        topic = current_topic or context["current_topic"]
        active_keywords = context["active_keywords"]

        # Apply topic relevance scoring
        scored_results = []
        for result in results:
            # Calculate topic relevance boost
            topic_boost = self._calculate_topic_relevance(
                result, topic, active_keywords
            )

            # Apply boost to relevance score
            boosted_score = min(1.0, result.relevance_score * topic_boost)

            # Update result with boosted score
            result.relevance_score = boosted_score
            result.search_type = "context_aware"

            scored_results.append(result)

        # Sort by boosted relevance
        scored_results.sort(key=lambda x: x.relevance_score, reverse=True)

        self.logger.info(
            f"Prioritized {len(results)} results for topic '{topic}' "
            f"with active keywords: {len(active_keywords)}"
        )

        return scored_results

    def get_topic_summary(
        self, conversation_id: str, limit: int = 20
    ) -> Dict[str, Any]:
        """
        Get topic summary for a conversation.

        Args:
            conversation_id: ID of conversation to analyze
            limit: Number of messages to analyze

        Returns:
            Dictionary with topic analysis
        """
        try:
            # Get recent messages
            messages = self.sqlite_manager.get_recent_messages(
                conversation_id, limit=limit
            )

            if not messages:
                return {"topic": "general", "keywords": [], "message_count": 0}

            # Combine all message content
            all_text = " ".join([msg.get("content", "") for msg in messages])

            # Analyze topics and keywords
            topic = self._classify_topic(all_text)
            keywords = list(self._extract_keywords(all_text))

            # Calculate topic distribution
            topic_distribution = {}
            for msg in messages:
                msg_topic = self._classify_topic(msg.get("content", ""))
                topic_distribution[msg_topic] = topic_distribution.get(msg_topic, 0) + 1

            return {
                "primary_topic": topic,
                "all_keywords": keywords,
                "message_count": len(messages),
                "topic_distribution": topic_distribution,
                "recent_focus": topic if len(messages) >= 5 else "general",
            }

        except Exception as e:
            self.logger.error(f"Failed to get topic summary: {e}")
            return {"topic": "general", "keywords": [], "message_count": 0}

    def suggest_related_topics(self, query: str, limit: int = 3) -> List[str]:
        """
        Suggest related topics based on query analysis.

        Args:
            query: Search query to analyze
            limit: Maximum number of suggestions

        Returns:
            List of suggested topic strings
        """
        query_topic = self._classify_topic(query)
        query_keywords = self._extract_keywords(query)

        # Find topics with overlapping keywords
        topic_scores = {}
        for topic, keywords in self.topic_keywords.items():
            if topic == query_topic:
                continue

            overlap = len(query_keywords & set(keywords))
            if overlap > 0:
                topic_scores[topic] = overlap

        # Sort by keyword overlap and return top suggestions
        suggested = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, _ in suggested[:limit]]

    def is_context_relevant(
        self, result: SearchResult, conversation_id: str, threshold: float = 0.3
    ) -> bool:
        """
        Check if a search result is relevant to current conversation context.

        Args:
            result: SearchResult to check
            conversation_id: Current conversation ID
            threshold: Minimum relevance threshold

        Returns:
            True if result is contextually relevant
        """
        context = self._get_current_context(conversation_id)

        # Calculate contextual relevance
        contextual_relevance = self._calculate_topic_relevance(
            result, context["current_topic"], context["active_keywords"]
        )

        # Adjust original score with contextual relevance
        adjusted_score = result.relevance_score * (contextual_relevance / 1.5)

        return adjusted_score >= threshold

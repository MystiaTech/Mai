"""
Timeline search implementation with date-range filtering and temporal analysis.

This module provides timeline-based search capabilities that allow filtering
conversations by date ranges, recency, and temporal proximity.
"""

import sys
import os
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import logging

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from .search_types import SearchResult, SearchQuery


class TimelineSearch:
    """
    Timeline search with date-range filtering and temporal search.

    Provides time-based search capabilities including date range filtering,
    temporal proximity search, and recency-based result weighting.
    """

    def __init__(self, sqlite_manager):
        """
        Initialize timeline search with SQLite manager.

        Args:
            sqlite_manager: SQLiteManager instance for temporal data access
        """
        self.sqlite_manager = sqlite_manager
        self.logger = logging.getLogger(__name__)

        # Compression awareness - conversations are compressed at different ages
        self.compression_tiers = {
            "recent": timedelta(days=7),  # Full detail
            "medium": timedelta(days=30),  # Key points
            "old": timedelta(days=90),  # Brief summary
            "archived": timedelta(days=365),  # Metadata only
        }

    def _get_compression_level(self, age: timedelta) -> str:
        """
        Determine compression level based on conversation age.

        Args:
            age: Age of the conversation

        Returns:
            Compression level string
        """
        if age <= self.compression_tiers["recent"]:
            return "full"
        elif age <= self.compression_tiers["medium"]:
            return "key_points"
        elif age <= self.compression_tiers["old"]:
            return "summary"
        else:
            return "metadata"

    def _calculate_recency_score(self, timestamp: datetime) -> float:
        """
        Calculate recency-based score boost.

        Args:
            timestamp: Message timestamp

        Returns:
            Recency boost factor (1.0 = no boost, >1.0 = recent)
        """
        now = datetime.utcnow()
        age = now - timestamp

        # Very recent (last 24 hours)
        if age <= timedelta(hours=24):
            return 1.5
        # Recent (last week)
        elif age <= timedelta(days=7):
            return 1.3
        # Semi-recent (last month)
        elif age <= timedelta(days=30):
            return 1.1
        # Older (no boost, slight penalty)
        else:
            return 0.9

    def _calculate_temporal_proximity_score(
        self, target_date: datetime, message_date: datetime
    ) -> float:
        """
        Calculate temporal proximity score for date-based search.

        Args:
            target_date: Target date to find conversations near
            message_date: Date of the message/conversation

        Returns:
            Proximity score (1.0 = exact match, decreasing with distance)
        """
        distance = abs(target_date - message_date)

        # Exact match
        if distance == timedelta(0):
            return 1.0

        # Within 1 day
        elif distance <= timedelta(days=1):
            return 0.9
        # Within 1 week
        elif distance <= timedelta(days=7):
            return 0.7
        # Within 1 month
        elif distance <= timedelta(days=30):
            return 0.5
        # Within 3 months
        elif distance <= timedelta(days=90):
            return 0.3
        # Older
        else:
            return 0.1

    def _create_timeline_result(
        self,
        conversation_id: str,
        message_id: str,
        content: str,
        timestamp: datetime,
        metadata: Dict[str, Any],
        temporal_score: float,
    ) -> SearchResult:
        """
        Create search result with temporal scoring.

        Args:
            conversation_id: ID of the conversation
            message_id: ID of the message
            content: Message content
            timestamp: Message timestamp
            metadata: Additional metadata
            temporal_score: Temporal relevance score

        Returns:
            SearchResult with timeline search type
        """
        # Generate snippet based on compression level
        age = datetime.utcnow() - timestamp
        compression_level = self._get_compression_level(age)

        if compression_level == "full":
            snippet = content[:300] + "..." if len(content) > 300 else content
        elif compression_level == "key_points":
            snippet = content[:150] + "..." if len(content) > 150 else content
        elif compression_level == "summary":
            snippet = content[:75] + "..." if len(content) > 75 else content
        else:  # metadata
            snippet = content[:50] + "..." if len(content) > 50 else content

        return SearchResult(
            conversation_id=conversation_id,
            message_id=message_id,
            content=content,
            relevance_score=temporal_score,
            snippet=snippet,
            timestamp=timestamp,
            metadata={
                **metadata,
                "age_days": age.days,
                "compression_level": compression_level,
                "temporal_score": temporal_score,
            },
            search_type="timeline",
        )

    def search_by_date_range(
        self, start: datetime, end: datetime, limit: int = 5
    ) -> List[SearchResult]:
        """
        Search conversations within a specific date range.

        Args:
            start: Start date (inclusive)
            end: End date (inclusive)
            limit: Maximum number of results to return

        Returns:
            List of search results within date range
        """
        if start >= end:
            self.logger.warning("Invalid date range: start must be before end")
            return []

        try:
            # Get conversations in date range from SQLite
            messages = self.sqlite_manager.get_messages_by_date_range(
                start, end, limit * 2
            )

            results = []
            for message in messages:
                # Calculate temporal relevance based on recency
                recency_score = self._calculate_recency_score(
                    message.get("timestamp", datetime.utcnow())
                )

                # Create search result
                result = self._create_timeline_result(
                    conversation_id=message.get("conversation_id", ""),
                    message_id=message.get("id", ""),
                    content=message.get("content", ""),
                    timestamp=message.get("timestamp", datetime.utcnow()),
                    metadata=message.get("metadata", {}),
                    temporal_score=recency_score,
                )
                results.append(result)

            # Sort by timestamp (most recent first) and limit
            results.sort(key=lambda x: x.timestamp, reverse=True)
            return results[:limit]

        except Exception as e:
            self.logger.error(f"Date range search failed: {e}")
            return []

    def search_near_date(
        self, target_date: datetime, days_range: int = 7, limit: int = 5
    ) -> List[SearchResult]:
        """
        Search for conversations near a specific date.

        Args:
            target_date: Target date to search around
            days_range: Number of days before/after to include
            limit: Maximum number of results to return

        Returns:
            List of search results temporally close to target
        """
        try:
            # Calculate date range around target
            start = target_date - timedelta(days=days_range)
            end = target_date + timedelta(days=days_range)

            # Get messages in extended range
            messages = self.sqlite_manager.get_messages_by_date_range(
                start, end, limit * 3
            )

            results = []
            for message in messages:
                # Calculate temporal proximity score
                proximity_score = self._calculate_temporal_proximity_score(
                    target_date, message.get("timestamp", datetime.utcnow())
                )

                # Create search result
                result = self._create_timeline_result(
                    conversation_id=message.get("conversation_id", ""),
                    message_id=message.get("id", ""),
                    content=message.get("content", ""),
                    timestamp=message.get("timestamp", datetime.utcnow()),
                    metadata=message.get("metadata", {}),
                    temporal_score=proximity_score,
                )
                results.append(result)

            # Sort by proximity score and limit
            results.sort(key=lambda x: x.relevance_score, reverse=True)
            return results[:limit]

        except Exception as e:
            self.logger.error(f"Near date search failed: {e}")
            return []

    def search_recent(self, days: int = 7, limit: int = 5) -> List[SearchResult]:
        """
        Search for recent conversations within specified days.

        Args:
            days: Number of recent days to search
            limit: Maximum number of results to return

        Returns:
            List of recent search results
        """
        end = datetime.utcnow()
        start = end - timedelta(days=days)

        return self.search_by_date_range(start, end, limit)

    def get_temporal_summary(
        self, conversation_id: Optional[str] = None, days: int = 30
    ) -> Dict[str, Any]:
        """
        Get temporal summary of conversations.

        Args:
            conversation_id: Specific conversation to analyze (None for all)
            days: Number of recent days to analyze

        Returns:
            Dictionary with temporal statistics
        """
        try:
            end = datetime.utcnow()
            start = end - timedelta(days=days)

            # Get messages in time range
            messages = self.sqlite_manager.get_messages_by_date_range(
                start,
                end,
                limit=1000,  # Get all for analysis
            )

            if conversation_id:
                messages = [
                    msg
                    for msg in messages
                    if msg.get("conversation_id") == conversation_id
                ]

            if not messages:
                return {
                    "total_messages": 0,
                    "date_range": f"{start.date()} to {end.date()}",
                    "daily_average": 0.0,
                    "peak_days": [],
                }

            # Analyze temporal patterns
            daily_counts = {}
            for message in messages:
                date = message.get("timestamp", datetime.utcnow()).date()
                daily_counts[date] = daily_counts.get(date, 0) + 1

            # Calculate statistics
            total_messages = len(messages)
            days_in_range = (end - start).days or 1
            daily_average = total_messages / days_in_range

            # Find peak activity days
            peak_days = sorted(daily_counts.items(), key=lambda x: x[1], reverse=True)[
                :5
            ]

            return {
                "total_messages": total_messages,
                "date_range": f"{start.date()} to {end.date()}",
                "days_analyzed": days_in_range,
                "daily_average": round(daily_average, 2),
                "peak_days": [
                    {"date": str(date), "count": count} for date, count in peak_days
                ],
                "compression_distribution": self._analyze_compression_distribution(
                    messages
                ),
            }

        except Exception as e:
            self.logger.error(f"Failed to get temporal summary: {e}")
            return {"error": str(e)}

    def _analyze_compression_distribution(
        self, messages: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """
        Analyze compression level distribution of messages.

        Args:
            messages: List of messages to analyze

        Returns:
            Dictionary with compression level counts
        """
        distribution = {"full": 0, "key_points": 0, "summary": 0, "metadata": 0}
        now = datetime.utcnow()

        for message in messages:
            timestamp = message.get("timestamp", now)
            age = now - timestamp
            level = self._get_compression_level(age)
            distribution[level] = distribution.get(level, 0) + 1

        return distribution

    def find_conversations_around_topic(
        self, topic_keywords: List[str], days_range: int = 30, limit: int = 5
    ) -> List[SearchResult]:
        """
        Find conversations around specific topic keywords within time range.

        Args:
            topic_keywords: Keywords related to the topic
            days_range: Number of days to search back
            limit: Maximum number of results

        Returns:
            List of search results with topic relevance
        """
        end = datetime.utcnow()
        start = end - timedelta(days=days_range)

        try:
            # Get messages in time range
            messages = self.sqlite_manager.get_messages_by_date_range(
                start, end, limit * 2
            )

            results = []
            for message in messages:
                content = message.get("content", "").lower()

                # Count keyword matches
                keyword_matches = sum(
                    1 for keyword in topic_keywords if keyword.lower() in content
                )

                if keyword_matches > 0:
                    # Calculate topic relevance score
                    topic_score = min(1.0, keyword_matches / len(topic_keywords))

                    # Combine with recency score
                    recency_score = self._calculate_recency_score(
                        message.get("timestamp", datetime.utcnow())
                    )

                    combined_score = topic_score * recency_score

                    result = self._create_timeline_result(
                        conversation_id=message.get("conversation_id", ""),
                        message_id=message.get("id", ""),
                        content=message.get("content", ""),
                        timestamp=message.get("timestamp", datetime.utcnow()),
                        metadata=message.get("metadata", {}),
                        temporal_score=combined_score,
                    )
                    result.metadata["keyword_matches"] = keyword_matches
                    results.append(result)

            # Sort by combined score and limit
            results.sort(key=lambda x: x.relevance_score, reverse=True)
            return results[:limit]

        except Exception as e:
            self.logger.error(f"Topic timeline search failed: {e}")
            return []

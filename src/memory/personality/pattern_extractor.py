"""
Pattern extraction system for personality learning.

This module extracts multi-dimensional patterns from conversations
including topics, sentiment, interaction patterns, temporal patterns,
and response styles.
"""

import re
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import statistics

# Import conversation models
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from models.conversation import Message, MessageRole, ConversationMetadata


@dataclass
class TopicPatterns:
    """Topic pattern analysis results."""

    frequent_topics: List[Tuple[str, float]] = field(default_factory=list)
    topic_diversity: float = 0.0
    topic_transitions: Dict[str, List[str]] = field(default_factory=dict)
    user_interests: List[str] = field(default_factory=list)
    confidence_score: float = 0.0


@dataclass
class SentimentPatterns:
    """Sentiment pattern analysis results."""

    overall_sentiment: float = 0.0  # -1 to 1 scale
    sentiment_variance: float = 0.0
    emotional_tone: str = "neutral"
    sentiment_keywords: Dict[str, int] = field(default_factory=dict)
    mood_fluctuations: List[Tuple[datetime, float]] = field(default_factory=list)
    confidence_score: float = 0.0


@dataclass
class InteractionPatterns:
    """Interaction pattern analysis results."""

    question_frequency: float = 0.0
    information_sharing: float = 0.0
    response_time_avg: float = 0.0
    conversation_balance: float = 0.0  # user vs assistant message ratio
    engagement_level: float = 0.0
    confidence_score: float = 0.0


@dataclass
class TemporalPatterns:
    """Temporal pattern analysis results."""

    preferred_times: List[Tuple[str, float]] = field(
        default_factory=list
    )  # (hour, frequency)
    day_of_week_patterns: Dict[str, float] = field(default_factory=dict)
    conversation_duration: float = 0.0
    session_frequency: float = 0.0
    time_based_style: Dict[str, str] = field(default_factory=dict)
    confidence_score: float = 0.0


@dataclass
class ResponseStylePatterns:
    """Response style pattern analysis results."""

    formality_level: float = 0.0  # 0 = casual, 1 = formal
    verbosity: float = 0.0  # average message length
    emoji_usage: float = 0.0
    humor_frequency: float = 0.0
    directness: float = 0.0  # how direct vs circumlocutory
    confidence_score: float = 0.0


class PatternExtractor:
    """
    Multi-dimensional pattern extraction from conversations.

    Extracts patterns across topics, sentiment, interaction styles,
    temporal preferences, and response styles with confidence scoring
    and stability tracking.
    """

    def __init__(self):
        """Initialize pattern extractor with analysis configurations."""
        self.logger = logging.getLogger(__name__)

        # Sentiment keyword dictionaries
        self.positive_words = {
            "good",
            "great",
            "excellent",
            "amazing",
            "wonderful",
            "fantastic",
            "love",
            "like",
            "enjoy",
            "happy",
            "pleased",
            "satisfied",
            "perfect",
            "awesome",
            "brilliant",
            "outstanding",
            "superb",
            "delightful",
        }

        self.negative_words = {
            "bad",
            "terrible",
            "awful",
            "horrible",
            "hate",
            "dislike",
            "angry",
            "sad",
            "frustrated",
            "disappointed",
            "annoyed",
            "upset",
            "worried",
            "concerned",
            "problem",
            "issue",
            "error",
            "wrong",
            "fail",
            "failed",
        }

        # Topic extraction keywords
        self.topic_indicators = {
            "technology": [
                "computer",
                "software",
                "code",
                "programming",
                "app",
                "system",
            ],
            "work": ["job", "career", "project", "task", "meeting", "deadline"],
            "personal": ["family", "friend", "relationship", "home", "life", "health"],
            "entertainment": ["movie", "music", "game", "book", "show", "play"],
            "learning": ["study", "learn", "course", "education", "knowledge", "skill"],
        }

        # Formality indicators
        self.formal_indicators = [
            "please",
            "thank",
            "regards",
            "sincerely",
            "would",
            "could",
        ]
        self.casual_indicators = ["hey", "yo", "sup", "lol", "omg", "btw", "idk"]

        # Pattern stability tracking
        self._pattern_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def extract_topic_patterns(
        self, conversations: List[Dict[str, Any]]
    ) -> TopicPatterns:
        """
        Extract topic patterns from conversations.

        Args:
            conversations: List of conversation dictionaries with messages

        Returns:
            TopicPatterns object with extracted topic information
        """
        try:
            self.logger.info("Extracting topic patterns from conversations")

            # Collect all text content
            all_text = []
            topic_transitions = defaultdict(list)
            last_topic = None

            for conv in conversations:
                messages = conv.get("messages", [])
                for msg in messages:
                    if msg.get("role") in ["user", "assistant"]:
                        content = msg.get("content", "").lower()
                        all_text.append(content)

                        # Extract current topic
                        current_topic = self._identify_main_topic(content)
                        if current_topic and last_topic and current_topic != last_topic:
                            topic_transitions[last_topic].append(current_topic)
                        last_topic = current_topic

            # Frequency analysis
            topic_counts = Counter()
            for text in all_text:
                topic = self._identify_main_topic(text)
                if topic:
                    topic_counts[topic] += 1

            # Calculate frequent topics
            total_topics = sum(topic_counts.values())
            frequent_topics = (
                [
                    (topic, count / total_topics)
                    for topic, count in topic_counts.most_common(10)
                ]
                if total_topics > 0
                else []
            )

            # Calculate topic diversity (Shannon entropy)
            topic_diversity = self._calculate_diversity(topic_counts)

            # Extract user interests (most frequent topics from user messages)
            user_interests = list(dict(frequent_topics[:5]).keys())

            # Calculate confidence score
            confidence = self._calculate_topic_confidence(
                topic_counts, len(all_text), frequent_topics
            )

            return TopicPatterns(
                frequent_topics=frequent_topics,
                topic_diversity=topic_diversity,
                topic_transitions=dict(topic_transitions),
                user_interests=user_interests,
                confidence_score=confidence,
            )

        except Exception as e:
            self.logger.error(f"Failed to extract topic patterns: {e}")
            return TopicPatterns(confidence_score=0.0)

    def extract_sentiment_patterns(
        self, conversations: List[Dict[str, Any]]
    ) -> SentimentPatterns:
        """
        Extract sentiment patterns from conversations.

        Args:
            conversations: List of conversation dictionaries with messages

        Returns:
            SentimentPatterns object with extracted sentiment information
        """
        try:
            self.logger.info("Extracting sentiment patterns from conversations")

            sentiment_scores = []
            sentiment_keywords = Counter()
            mood_fluctuations = []

            for conv in conversations:
                messages = conv.get("messages", [])
                for msg in messages:
                    if msg.get("role") in ["user", "assistant"]:
                        content = msg.get("content", "").lower()

                        # Calculate sentiment score
                        score = self._calculate_sentiment_score(content)
                        sentiment_scores.append(score)

                        # Track sentiment keywords
                        for word in self.positive_words:
                            if word in content:
                                sentiment_keywords[f"positive_{word}"] += 1
                        for word in self.negative_words:
                            if word in content:
                                sentiment_keywords[f"negative_{word}"] += 1

                        # Track mood over time
                        if "timestamp" in msg:
                            timestamp = msg["timestamp"]
                            if isinstance(timestamp, str):
                                timestamp = datetime.fromisoformat(
                                    timestamp.replace("Z", "+00:00")
                                )
                            mood_fluctuations.append((timestamp, score))

            # Calculate overall sentiment
            overall_sentiment = (
                statistics.mean(sentiment_scores) if sentiment_scores else 0.0
            )

            # Calculate sentiment variance
            sentiment_variance = (
                statistics.variance(sentiment_scores)
                if len(sentiment_scores) > 1
                else 0.0
            )

            # Determine emotional tone
            emotional_tone = self._classify_emotional_tone(overall_sentiment)

            # Calculate confidence score
            confidence = self._calculate_sentiment_confidence(
                sentiment_scores, len(sentiment_keywords)
            )

            return SentimentPatterns(
                overall_sentiment=overall_sentiment,
                sentiment_variance=sentiment_variance,
                emotional_tone=emotional_tone,
                sentiment_keywords=dict(sentiment_keywords),
                mood_fluctuations=mood_fluctuations,
                confidence_score=confidence,
            )

        except Exception as e:
            self.logger.error(f"Failed to extract sentiment patterns: {e}")
            return SentimentPatterns(confidence_score=0.0)

    def extract_interaction_patterns(
        self, conversations: List[Dict[str, Any]]
    ) -> InteractionPatterns:
        """
        Extract interaction patterns from conversations.

        Args:
            conversations: List of conversation dictionaries with messages

        Returns:
            InteractionPatterns object with extracted interaction information
        """
        try:
            self.logger.info("Extracting interaction patterns from conversations")

            question_count = 0
            info_sharing_count = 0
            response_times = []
            user_messages = 0
            assistant_messages = 0
            engagement_indicators = []

            for conv in conversations:
                messages = conv.get("messages", [])
                prev_timestamp = None

                for i, msg in enumerate(messages):
                    role = msg.get("role")
                    content = msg.get("content", "").lower()

                    # Count questions
                    if "?" in content and role == "user":
                        question_count += 1

                    # Count information sharing
                    info_sharing_indicators = [
                        "because",
                        "since",
                        "due to",
                        "reason is",
                        "explanation",
                    ]
                    if any(
                        indicator in content for indicator in info_sharing_indicators
                    ):
                        info_sharing_count += 1

                    # Track message counts for balance
                    if role == "user":
                        user_messages += 1
                    elif role == "assistant":
                        assistant_messages += 1

                    # Calculate response times
                    if prev_timestamp and "timestamp" in msg:
                        try:
                            curr_time = msg["timestamp"]
                            if isinstance(curr_time, str):
                                curr_time = datetime.fromisoformat(
                                    curr_time.replace("Z", "+00:00")
                                )

                            time_diff = (curr_time - prev_timestamp).total_seconds()
                            if 0 < time_diff < 3600:  # Within reasonable range
                                response_times.append(time_diff)
                        except Exception:
                            pass

                    # Track engagement indicators
                    engagement_words = [
                        "interesting",
                        "tell me more",
                        "fascinating",
                        "cool",
                        "wow",
                    ]
                    if any(word in content for word in engagement_words):
                        engagement_indicators.append(1)
                    else:
                        engagement_indicators.append(0)

                    prev_timestamp = msg.get("timestamp")
                    if isinstance(prev_timestamp, str):
                        prev_timestamp = datetime.fromisoformat(
                            prev_timestamp.replace("Z", "+00:00")
                        )

            # Calculate metrics
            total_messages = user_messages + assistant_messages
            question_frequency = question_count / max(user_messages, 1)
            information_sharing = info_sharing_count / max(total_messages, 1)
            response_time_avg = (
                statistics.mean(response_times) if response_times else 0.0
            )
            conversation_balance = user_messages / max(total_messages, 1)
            engagement_level = (
                statistics.mean(engagement_indicators) if engagement_indicators else 0.0
            )

            # Calculate confidence score
            confidence = self._calculate_interaction_confidence(
                total_messages, len(response_times), question_count
            )

            return InteractionPatterns(
                question_frequency=question_frequency,
                information_sharing=information_sharing,
                response_time_avg=response_time_avg,
                conversation_balance=conversation_balance,
                engagement_level=engagement_level,
                confidence_score=confidence,
            )

        except Exception as e:
            self.logger.error(f"Failed to extract interaction patterns: {e}")
            return InteractionPatterns(confidence_score=0.0)

    def extract_temporal_patterns(
        self, conversations: List[Dict[str, Any]]
    ) -> TemporalPatterns:
        """
        Extract temporal patterns from conversations.

        Args:
            conversations: List of conversation dictionaries with messages

        Returns:
            TemporalPatterns object with extracted temporal information
        """
        try:
            self.logger.info("Extracting temporal patterns from conversations")

            hour_counts = Counter()
            day_counts = Counter()
            conversation_durations = []
            session_start_times = []

            for conv in conversations:
                messages = conv.get("messages", [])
                if not messages:
                    continue

                # Track conversation duration
                timestamps = []
                for msg in messages:
                    if "timestamp" in msg:
                        try:
                            timestamp = msg["timestamp"]
                            if isinstance(timestamp, str):
                                timestamp = datetime.fromisoformat(
                                    timestamp.replace("Z", "+00:00")
                                )
                            timestamps.append(timestamp)
                        except Exception:
                            continue

                if timestamps:
                    # Calculate duration
                    duration = (
                        max(timestamps) - min(timestamps)
                    ).total_seconds() / 60  # minutes
                    conversation_durations.append(duration)

                    # Count hour and day patterns
                    for timestamp in timestamps:
                        hour_counts[timestamp.hour] += 1
                        day_counts[timestamp.strftime("%A")] += 1

                    # Track session start time
                    session_start_times.append(min(timestamps))

            # Calculate preferred times
            total_hours = sum(hour_counts.values())
            preferred_times = (
                [
                    (str(hour), count / total_hours)
                    for hour, count in hour_counts.most_common(5)
                ]
                if total_hours > 0
                else []
            )

            # Calculate day of week patterns
            total_days = sum(day_counts.values())
            day_of_week_patterns = (
                {day: count / total_days for day, count in day_counts.items()}
                if total_days > 0
                else {}
            )

            # Calculate other metrics
            avg_duration = (
                statistics.mean(conversation_durations)
                if conversation_durations
                else 0.0
            )

            # Calculate session frequency (sessions per day)
            if session_start_times:
                time_span = (
                    max(session_start_times) - min(session_start_times)
                ).days + 1
                session_frequency = len(session_start_times) / max(time_span, 1)
            else:
                session_frequency = 0.0

            # Time-based style analysis
            time_based_style = self._analyze_time_based_styles(conversations)

            # Calculate confidence score
            confidence = self._calculate_temporal_confidence(
                len(conversations), total_hours, len(session_start_times)
            )

            return TemporalPatterns(
                preferred_times=preferred_times,
                day_of_week_patterns=day_of_week_patterns,
                conversation_duration=avg_duration,
                session_frequency=session_frequency,
                time_based_style=time_based_style,
                confidence_score=confidence,
            )

        except Exception as e:
            self.logger.error(f"Failed to extract temporal patterns: {e}")
            return TemporalPatterns(confidence_score=0.0)

    def extract_response_style_patterns(
        self, conversations: List[Dict[str, Any]]
    ) -> ResponseStylePatterns:
        """
        Extract response style patterns from conversations.

        Args:
            conversations: List of conversation dictionaries with messages

        Returns:
            ResponseStylePatterns object with extracted response style information
        """
        try:
            self.logger.info("Extracting response style patterns from conversations")

            message_lengths = []
            formality_scores = []
            emoji_counts = []
            humor_indicators = []
            directness_scores = []

            for conv in conversations:
                messages = conv.get("messages", [])
                for msg in messages:
                    if msg.get("role") in ["user", "assistant"]:
                        content = msg.get("content", "")

                        # Message length (verbosity)
                        message_lengths.append(len(content.split()))

                        # Formality level
                        formality = self._calculate_formality(content)
                        formality_scores.append(formality)

                        # Emoji usage
                        emoji_count = len(
                            re.findall(
                                r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]",
                                content,
                            )
                        )
                        emoji_counts.append(emoji_count)

                        # Humor frequency
                        humor_words = [
                            "lol",
                            "haha",
                            "funny",
                            "joke",
                            "hilarious",
                            "😂",
                            "😄",
                        ]
                        humor_indicators.append(
                            1
                            if any(word in content.lower() for word in humor_words)
                            else 0
                        )

                        # Directness (simple vs complex sentences)
                        directness = self._calculate_directness(content)
                        directness_scores.append(directness)

            # Calculate averages
            verbosity = statistics.mean(message_lengths) if message_lengths else 0.0
            formality_level = (
                statistics.mean(formality_scores) if formality_scores else 0.0
            )
            emoji_usage = statistics.mean(emoji_counts) if emoji_counts else 0.0
            humor_frequency = (
                statistics.mean(humor_indicators) if humor_indicators else 0.0
            )
            directness = (
                statistics.mean(directness_scores) if directness_scores else 0.0
            )

            # Calculate confidence score
            confidence = self._calculate_style_confidence(
                len(message_lengths), len(formality_scores)
            )

            return ResponseStylePatterns(
                formality_level=formality_level,
                verbosity=verbosity,
                emoji_usage=emoji_usage,
                humor_frequency=humor_frequency,
                directness=directness,
                confidence_score=confidence,
            )

        except Exception as e:
            self.logger.error(f"Failed to extract response style patterns: {e}")
            return ResponseStylePatterns(confidence_score=0.0)

    def _identify_main_topic(self, text: str) -> Optional[str]:
        """Identify the main topic of a text snippet."""
        topic_scores = defaultdict(int)

        for topic, keywords in self.topic_indicators.items():
            for keyword in keywords:
                if keyword in text:
                    topic_scores[topic] += 1

        if topic_scores:
            return max(topic_scores, key=topic_scores.get)
        return None

    def _calculate_diversity(self, counts: Counter) -> float:
        """Calculate Shannon entropy diversity."""
        total = sum(counts.values())
        if total == 0:
            return 0.0

        entropy = 0.0
        for count in counts.values():
            probability = count / total
            entropy -= probability * (
                probability and statistics.log(probability, 2) or 0
            )

        return entropy

    def _calculate_sentiment_score(self, text: str) -> float:
        """Calculate sentiment score for text (-1 to 1)."""
        positive_count = sum(1 for word in self.positive_words if word in text)
        negative_count = sum(1 for word in self.negative_words if word in text)

        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words == 0:
            return 0.0

        return (positive_count - negative_count) / total_sentiment_words

    def _classify_emotional_tone(self, sentiment: float) -> str:
        """Classify emotional tone from sentiment score."""
        if sentiment > 0.3:
            return "positive"
        elif sentiment < -0.3:
            return "negative"
        else:
            return "neutral"

    def _calculate_formality(self, text: str) -> float:
        """Calculate formality level (0 = casual, 1 = formal)."""
        formal_count = sum(1 for word in self.formal_indicators if word in text.lower())
        casual_count = sum(1 for word in self.casual_indicators if word in text.lower())

        # Base formality on presence of formal indicators and absence of casual ones
        if formal_count > 0 and casual_count == 0:
            return 0.8
        elif formal_count == 0 and casual_count > 0:
            return 0.2
        elif formal_count > casual_count:
            return 0.6
        elif casual_count > formal_count:
            return 0.4
        else:
            return 0.5

    def _calculate_directness(self, text: str) -> float:
        """Calculate directness (0 = circumlocutory, 1 = direct)."""
        # Simple heuristic: shorter sentences and fewer subordinate clauses are more direct
        sentences = text.split(".")
        if not sentences:
            return 0.5

        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        subordinate_indicators = [
            "because",
            "although",
            "however",
            "therefore",
            "meanwhile",
        ]
        subordinate_count = sum(
            1 for indicator in subordinate_indicators if indicator in text.lower()
        )

        # Directness decreases with longer sentences and more subordinate clauses
        directness = 1.0 - (avg_sentence_length / 50.0) - (subordinate_count * 0.1)
        return max(0.0, min(1.0, directness))

    def _analyze_time_based_styles(
        self, conversations: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Analyze how communication style changes by time."""
        time_styles = {}

        for conv in conversations:
            messages = conv.get("messages", [])
            for msg in messages:
                if "timestamp" in msg:
                    try:
                        timestamp = msg["timestamp"]
                        if isinstance(timestamp, str):
                            timestamp = datetime.fromisoformat(
                                timestamp.replace("Z", "+00:00")
                            )

                        hour = timestamp.hour
                        content = msg.get("content", "").lower()

                        # Simple style classification by time
                        if 6 <= hour < 12:  # Morning
                            style = (
                                "morning_formal"
                                if any(
                                    word in self.formal_indicators
                                    for word in self.formal_indicators
                                    if word in content
                                )
                                else "morning_casual"
                            )
                        elif 12 <= hour < 18:  # Afternoon
                            style = (
                                "afternoon_direct"
                                if len(content.split()) < 10
                                else "afternoon_detailed"
                            )
                        elif 18 <= hour < 22:  # Evening
                            style = "evening_relaxed"
                        else:  # Night
                            style = "night_concise"

                        time_styles[f"{hour}:00"] = style
                    except Exception:
                        continue

        return time_styles

    def _calculate_topic_confidence(
        self, topic_counts: Counter, total_messages: int, frequent_topics: List
    ) -> float:
        """Calculate confidence score for topic patterns."""
        if total_messages == 0:
            return 0.0

        # Confidence based on topic clarity and frequency
        topic_coverage = sum(count for _, count in frequent_topics) / total_messages
        topic_variety = len(topic_counts) / max(total_messages, 1)

        return min(1.0, (topic_coverage + topic_variety) / 2)

    def _calculate_sentiment_confidence(
        self, sentiment_scores: List[float], keyword_count: int
    ) -> float:
        """Calculate confidence score for sentiment patterns."""
        if not sentiment_scores:
            return 0.0

        # Confidence based on consistency and keyword evidence
        sentiment_consistency = 1.0 - (
            statistics.stdev(sentiment_scores) if len(sentiment_scores) > 1 else 0.0
        )
        keyword_evidence = min(1.0, keyword_count / len(sentiment_scores))

        return (sentiment_consistency + keyword_evidence) / 2

    def _calculate_interaction_confidence(
        self, total_messages: int, response_times: int, questions: int
    ) -> float:
        """Calculate confidence score for interaction patterns."""
        if total_messages == 0:
            return 0.0

        # Confidence based on data completeness
        message_coverage = min(
            1.0, total_messages / 10
        )  # More messages = higher confidence
        response_coverage = min(1.0, response_times / max(total_messages // 2, 1))
        question_coverage = min(1.0, questions / max(total_messages // 10, 1))

        return (message_coverage + response_coverage + question_coverage) / 3

    def _calculate_temporal_confidence(
        self, conversations: int, hour_data: int, sessions: int
    ) -> float:
        """Calculate confidence score for temporal patterns."""
        if conversations == 0:
            return 0.0

        # Confidence based on temporal data spread
        conversation_coverage = min(1.0, conversations / 5)
        hour_coverage = min(1.0, hour_data / 24)
        session_coverage = min(1.0, sessions / 3)

        return (conversation_coverage + hour_coverage + session_coverage) / 3

    def _calculate_style_confidence(self, messages: int, formality_data: int) -> float:
        """Calculate confidence score for style patterns."""
        if messages == 0:
            return 0.0

        # Confidence based on style data completeness
        message_coverage = min(1.0, messages / 10)
        formality_coverage = min(1.0, formality_data / max(messages, 1))

        return (message_coverage + formality_coverage) / 2

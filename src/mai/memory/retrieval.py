"""
Context Retrieval Implementation for Mai Memory System

Provides intelligent context retrieval with multi-faceted search,
adaptive weighting, and strategic context placement to prevent
"lost in the middle" problems.
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass

# Import Mai components
try:
    from src.mai.memory.storage import MemoryStorage, VectorSearchError, DatabaseConnectionError
    from src.mai.models.conversation import Conversation
    from src.mai.models.memory import (
        SearchQuery,
        RetrievalResult,
        MemoryContext,
        ContextWeight,
        ConversationType,
        RelevanceType,
        ConversationPattern,
        ContextPlacement,
    )
    from src.mai.core.config import get_config
    from src.mai.core.exceptions import MaiError, ContextError, create_error_context
except ImportError as e:
    # Handle missing dependencies gracefully
    logging.warning(f"Could not import Mai components: {e}")
    MemoryStorage = None
    SearchQuery = None
    RetrievalResult = None
    MemoryContext = None
    ContextWeight = None
    ConversationType = None
    RelevanceType = None
    ConversationPattern = None
    ContextPlacement = None
    MaiError = Exception
    ContextError = Exception
    VectorSearchError = Exception
    DatabaseConnectionError = Exception

    def create_error_context(component: str, operation: str, **data):
        return {"component": component, "operation": operation, "data": data}

    def get_config():
        return {"memory": {"weights": {"pattern": 0.1}}}


logger = logging.getLogger(__name__)


class ContextRetrievalError(ContextError):
    """Context retrieval specific errors."""

    def __init__(self, message: str, query: str = None, **kwargs):
        context = create_error_context(
            component="context_retrieval", operation="retrieve_context", query=query, **kwargs
        )
        super().__init__(message, context=context)
        self.query = query


class ContextRetriever:
    """
    Intelligent context retrieval system with multi-faceted search.

    Combines semantic similarity, keyword matching, recency weighting,
    and user pattern analysis to provide comprehensive, relevant context
    while preventing information overload and "lost in the middle" issues.
    """

    def __init__(self, storage: Optional[MemoryStorage] = None):
        """
        Initialize context retriever with storage and configuration.

        Args:
            storage: MemoryStorage instance (creates default if None)
        """
        self.storage = storage or MemoryStorage()
        self.config = get_config()

        # Load memory configuration
        self.memory_config = self.config.memory

        # Initialize search weights for different conversation types
        self._init_conversation_weights()

        # Pattern extraction cache
        self._pattern_cache = {}

        logger.info("ContextRetriever initialized with multi-faceted search")

    def _init_conversation_weights(self) -> None:
        """Initialize adaptive weights for different conversation types."""
        # Default weights from config
        self.default_weights = ContextWeight(
            semantic=self.memory_config.semantic_weight,
            keyword=self.memory_config.keyword_weight,
            recency=self.memory_config.recency_weight,
            user_pattern=self.memory_config.user_pattern_weight,
        )

        # Conversation type-specific weight adjustments
        self.type_weights = {
            ConversationType.TECHNICAL: ContextWeight(
                semantic=0.5, keyword=0.4, recency=0.05, user_pattern=0.05
            ),
            ConversationType.PERSONAL: ContextWeight(
                semantic=0.3, keyword=0.2, recency=0.3, user_pattern=0.2
            ),
            ConversationType.PLANNING: ContextWeight(
                semantic=0.4, keyword=0.2, recency=0.2, user_pattern=0.2
            ),
            ConversationType.QUESTION: ContextWeight(
                semantic=0.6, keyword=0.3, recency=0.05, user_pattern=0.05
            ),
            ConversationType.CREATIVE: ContextWeight(
                semantic=0.35, keyword=0.15, recency=0.2, user_pattern=0.3
            ),
            ConversationType.ANALYSIS: ContextWeight(
                semantic=0.45, keyword=0.35, recency=0.1, user_pattern=0.1
            ),
            ConversationType.GENERAL: self.default_weights,
        }

    def retrieve_context(self, query: SearchQuery) -> MemoryContext:
        """
        Retrieve comprehensive context for a search query.

        Args:
            query: SearchQuery with search parameters

        Returns:
            MemoryContext with retrieved conversations and metadata

        Raises:
            ContextRetrievalError: If retrieval fails
        """
        try:
            logger.info(f"Retrieving context for query: {query.text[:100]}...")

            # Detect conversation type if not provided
            detected_type = query.conversation_type or self._detect_conversation_type(query.text)
            query.conversation_type = detected_type

            # Get appropriate weights for this conversation type
            weights = self._get_adaptive_weights(detected_type, query.weights)

            # Perform multi-faceted search
            results = self._perform_multi_faceted_search(query, weights)

            # Rank and filter results
            ranked_results = self._rank_results(results, weights, query)

            # Apply context placement strategy
            final_results = self._apply_context_placement(ranked_results, query)

            # Create memory context
            context = MemoryContext(
                current_query=query,
                relevant_conversations=final_results,
                patterns=self._extract_patterns(final_results),
                metadata={
                    "weights_applied": weights.dict(),
                    "conversation_type": detected_type.value,
                    "search_facets": self._get_active_facets(query),
                },
            )

            # Set computed fields
            context.total_conversations = len(final_results)
            context.total_tokens = self._estimate_tokens(final_results)
            context.applied_weights = weights.dict()
            context.conversation_type_detected = detected_type

            logger.info(
                f"Retrieved {context.total_conversations} conversations, ~{context.total_tokens} tokens"
            )
            return context

        except Exception as e:
            raise ContextRetrievalError(
                message=f"Context retrieval failed: {e}", query=query.text, error_details=str(e)
            ) from e

    def _detect_conversation_type(self, text: str) -> ConversationType:
        """
        Detect conversation type from query text.

        Args:
            text: Query text to analyze

        Returns:
            Detected ConversationType
        """
        text_lower = text.lower()

        # Technical indicators
        technical_keywords = [
            "code",
            "function",
            "class",
            "algorithm",
            "debug",
            "implement",
            "api",
            "database",
            "server",
            "python",
            "javascript",
            "react",
        ]
        if any(keyword in text_lower for keyword in technical_keywords):
            return ConversationType.TECHNICAL

        # Question indicators
        question_indicators = ["?", "how", "what", "why", "when", "where", "which"]
        if text_lower.strip().endswith("?") or any(
            indicator in text_lower.split()[:3] for indicator in question_indicators
        ):
            return ConversationType.QUESTION

        # Planning indicators
        planning_keywords = [
            "plan",
            "schedule",
            "deadline",
            "task",
            "project",
            "goal",
            "organize",
            "implement",
            "roadmap",
        ]
        if any(keyword in text_lower for keyword in planning_keywords):
            return ConversationType.PLANNING

        # Creative indicators
        creative_keywords = [
            "create",
            "design",
            "write",
            "imagine",
            "brainstorm",
            "idea",
            "concept",
            "story",
            "novel",
            "art",
            "creative",
        ]
        if any(keyword in text_lower for keyword in creative_keywords):
            return ConversationType.CREATIVE

        # Analysis indicators
        analysis_keywords = [
            "analyze",
            "compare",
            "evaluate",
            "review",
            "assess",
            "examine",
            "pros",
            "cons",
            "advantages",
            "disadvantages",
        ]
        if any(keyword in text_lower for keyword in analysis_keywords):
            return ConversationType.ANALYSIS

        # Personal indicators (check last)
        personal_keywords = [
            "i feel",
            "i think",
            "my opinion",
            "personally",
            "experience",
            "remember",
            "preference",
            "favorite",
        ]
        if any(keyword in text_lower for keyword in personal_keywords):
            return ConversationType.PERSONAL

        # Default to general
        return ConversationType.GENERAL

    def _get_adaptive_weights(
        self, conv_type: ConversationType, overrides: Dict[str, float]
    ) -> ContextWeight:
        """
        Get adaptive weights for conversation type with optional overrides.

        Args:
            conv_type: Type of conversation
            overrides: Weight overrides from query

        Returns:
            ContextWeight with applied overrides
        """
        # Start with type-specific weights
        base_weights = self.type_weights.get(conv_type, self.default_weights)

        # Apply overrides
        if overrides:
            weight_dict = base_weights.dict()
            weight_dict.update(overrides)
            return ContextWeight(**weight_dict)

        return base_weights

    def _perform_multi_faceted_search(
        self, query: SearchQuery, weights: ContextWeight
    ) -> List[RetrievalResult]:
        """
        Perform multi-faceted search combining different search methods.

        Args:
            query: Search query with parameters
            weights: Search weights to apply

        Returns:
            List of retrieval results from different facets
        """
        all_results = []

        # Semantic similarity search
        if query.include_semantic and weights.semantic > 0:
            semantic_results = self._semantic_search(query)
            all_results.extend(semantic_results)
            logger.debug(f"Semantic search found {len(semantic_results)} results")

        # Keyword matching search
        if query.include_keywords and weights.keyword > 0:
            keyword_results = self._keyword_search(query)
            all_results.extend(keyword_results)
            logger.debug(f"Keyword search found {len(keyword_results)} results")

        # Recency-based search
        if query.include_recency and weights.recency > 0:
            recency_results = self._recency_search(query)
            all_results.extend(recency_results)
            logger.debug(f"Recency search found {len(recency_results)} results")

        # Pattern-based search
        if query.include_patterns and weights.pattern > 0:
            pattern_results = self._pattern_search(query)
            all_results.extend(pattern_results)
            logger.debug(f"Pattern search found {len(pattern_results)} results")

        return all_results

    def _semantic_search(self, query: SearchQuery) -> List[RetrievalResult]:
        """Perform semantic similarity search using vector embeddings."""
        try:
            # Use storage's search_conversations method
            results = self.storage.search_conversations(
                query=query.text, limit=query.max_results, include_content=True
            )

            # Convert to RetrievalResult objects
            semantic_results = []
            for result in results:
                retrieval_result = RetrievalResult(
                    conversation_id=result["conversation_id"],
                    title=result["title"],
                    similarity_score=result["similarity_score"],
                    relevance_type=RelevanceType.SEMANTIC,
                    excerpt=result["matched_message"]["content"][:500],
                    context_type=ConversationType.GENERAL,  # Will be refined later
                    matched_message_id=result.get("message_id"),
                    semantic_score=result["similarity_score"],
                )
                semantic_results.append(retrieval_result)

            return semantic_results

        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")
            return []

    def _keyword_search(self, query: SearchQuery) -> List[RetrievalResult]:
        """Perform keyword matching search."""
        try:
            # Extract keywords from query
            keywords = self._extract_keywords(query.text)
            if not keywords:
                return []

            # Get all conversations and search for keywords
            conversations = self.storage.get_conversation_list(limit=50)
            keyword_results = []

            for conv in conversations:
                # Get full conversation for content search
                full_conv = self.storage.retrieve_conversation(conv["id"])
                if not full_conv:
                    continue

                # Check keyword matches
                content_text = " ".join([msg["content"] for msg in full_conv["messages"]]).lower()
                keyword_matches = sum(1 for kw in keywords if kw.lower() in content_text)

                if keyword_matches > 0:
                    # Calculate keyword score based on match density
                    keyword_score = min(keyword_matches / len(keywords), 1.0)

                    retrieval_result = RetrievalResult(
                        conversation_id=conv["id"],
                        title=conv["title"],
                        similarity_score=keyword_score,
                        relevance_type=RelevanceType.KEYWORD,
                        excerpt=self._create_keyword_excerpt(content_text, keywords, 300),
                        keyword_score=keyword_score,
                    )
                    keyword_results.append(retrieval_result)

            return sorted(keyword_results, key=lambda x: x.keyword_score, reverse=True)[
                : query.max_results
            ]

        except Exception as e:
            logger.warning(f"Keyword search failed: {e}")
            return []

    def _recency_search(self, query: SearchQuery) -> List[RetrievalResult]:
        """Perform recency-based search for recent conversations."""
        try:
            # Get recent conversations
            conversations = self.storage.get_conversation_list(limit=query.max_results)

            recency_results = []
            now = datetime.now()

            for i, conv in enumerate(conversations):
                # Calculate recency score (newer = higher score)
                try:
                    updated_time = datetime.fromisoformat(conv["updated_at"].replace("Z", "+00:00"))
                    days_old = (now - updated_time).days

                    # Exponential decay: recent conversations get much higher scores
                    recency_score = max(0, 1.0 - (days_old / 30.0))  # 30-day window

                    retrieval_result = RetrievalResult(
                        conversation_id=conv["id"],
                        title=conv["title"],
                        similarity_score=recency_score,
                        relevance_type=RelevanceType.RECENCY,
                        excerpt=f"Recent conversation from {days_old} days ago",
                        recency_score=recency_score,
                    )
                    recency_results.append(retrieval_result)

                except (ValueError, KeyError) as e:
                    logger.debug(f"Could not parse timestamp for conversation {conv['id']}: {e}")
                    continue

            return sorted(recency_results, key=lambda x: x.recency_score, reverse=True)

        except Exception as e:
            logger.warning(f"Recency search failed: {e}")
            return []

    def _pattern_search(self, query: SearchQuery) -> List[RetrievalResult]:
        """Perform pattern-based search using stored user patterns."""
        try:
            if not self.storage:
                logger.warning("Storage not available for pattern search")
                return []

            # Load user patterns from storage
            user_patterns = self._load_user_patterns()

            if not user_patterns:
                logger.debug("No user patterns found for pattern search")
                return []

            # Perform pattern matching against query
            pattern_results = []

            for pattern_name, pattern_data in user_patterns.items():
                pattern_score = self._calculate_pattern_match_score(query.query_text, pattern_data)

                if pattern_score > 0.3:  # Threshold for pattern relevance
                    # Get associated conversations for this pattern
                    pattern_conversations = self._get_pattern_conversations(
                        pattern_name, pattern_data
                    )

                    for conversation in pattern_conversations:
                        result = RetrievalResult(
                            id=f"pattern_{pattern_name}_{conversation.id}",
                            conversation_id=conversation.id,
                            excerpt=conversation.summary or "",
                            relevance_score=pattern_score,
                            pattern_score=pattern_score,
                            timestamp=conversation.updated_at or conversation.created_at,
                            source="pattern_match",
                            content_type="pattern_match",
                            metadata={
                                "pattern_name": pattern_name,
                                "pattern_type": pattern_data.get("type", "keyword"),
                                "match_score": pattern_score,
                                "pattern_frequency": pattern_data.get("frequency", 0),
                            },
                        )
                        pattern_results.append(result)

            # Sort by pattern score and limit results
            pattern_results.sort(key=lambda x: x.pattern_score, reverse=True)

            logger.info(
                f"Pattern search found {len(pattern_results)} results from {len(user_patterns)} patterns"
            )
            return pattern_results[:50]  # Limit to prevent overwhelming results

        except Exception as e:
            logger.warning(f"Pattern search failed: {e}")
            return []

    def _rank_results(
        self, results: List[RetrievalResult], weights: ContextWeight, query: SearchQuery
    ) -> List[RetrievalResult]:
        """
        Rank and combine results from different search facets.

        Args:
            results: Raw results from all search facets
            weights: Search weights to apply
            query: Original search query

        Returns:
            Ranked and deduplicated results
        """
        # Group by conversation_id
        conversation_groups = {}
        for result in results:
            conv_id = result.conversation_id
            if conv_id not in conversation_groups:
                conversation_groups[conv_id] = []
            conversation_groups[conv_id].append(result)

        # Combine scores for each conversation
        combined_results = []
        for conv_id, group_results in conversation_groups.items():
            # Calculate weighted score
            weighted_score = 0.0
            best_semantic = 0.0
            best_keyword = 0.0
            best_recency = 0.0
            best_pattern = 0.0
            best_excerpt = ""

            for result in group_results:
                # Apply weights to facet scores
                if result.semantic_score is not None:
                    weighted_score += result.semantic_score * weights.semantic
                    best_semantic = max(best_semantic, result.semantic_score)
                if result.keyword_score is not None:
                    weighted_score += result.keyword_score * weights.keyword
                    best_keyword = max(best_keyword, result.keyword_score)
                if result.recency_score is not None:
                    weighted_score += result.recency_score * weights.recency
                    best_recency = max(best_recency, result.recency_score)
                if result.pattern_score is not None:
                    weighted_score += result.pattern_score * weights.pattern
                    best_pattern = max(best_pattern, result.pattern_score)

                # Keep the best excerpt
                if len(result.excerpt) > len(best_excerpt):
                    best_excerpt = result.excerpt

            # Create combined result
            combined_result = RetrievalResult(
                conversation_id=conv_id,
                title=group_results[0].title,
                similarity_score=min(weighted_score, 1.0),  # Cap at 1.0
                relevance_type=RelevanceType.HYBRID,
                excerpt=best_excerpt,
                semantic_score=best_semantic,
                keyword_score=best_keyword,
                recency_score=best_recency,
                pattern_score=best_pattern,
            )
            combined_results.append(combined_result)

        # Sort by combined score
        return sorted(combined_results, key=lambda x: x.similarity_score, reverse=True)

    def _apply_context_placement(
        self, results: List[RetrievalResult], query: SearchQuery
    ) -> List[RetrievalResult]:
        """
        Apply strategic context placement to prevent "lost in the middle".

        Uses a sophisticated algorithm to ensure important information remains
        visible throughout the context window, preventing degradation of
        information quality in the middle sections.

        Args:
            results: Ranked search results
            query: Search query for token limits

        Returns:
            Results with strategic ordering to maximize information retention
        """
        if not results:
            return results

        # Estimate token usage with better accuracy
        estimated_tokens = self._estimate_tokens(results)
        if estimated_tokens <= query.max_tokens:
            return self._optimize_ordering(results)  # Still optimize ordering even if all fit

        # Strategic placement algorithm
        return self._strategic_placement(results, query)

    def _strategic_placement(
        self, results: List[RetrievalResult], query: SearchQuery
    ) -> List[RetrievalResult]:
        """
        Implement strategic placement to prevent information loss.

        Strategy:
        1. Prime positions (first 20%) - highest priority items
        2. Middle reinforcement (40%-60%) - key reinforcing items
        3. Distributed placement - spread important items throughout
        4. Token-aware selection - respect context limits
        """
        if not results:
            return results

        max_tokens = query.max_tokens
        target_prime_tokens = int(max_tokens * 0.2)  # 20% for prime position
        target_middle_tokens = int(max_tokens * 0.2)  # 20% for middle reinforcement

        # Categorize results by quality and importance
        prime_results = []  # Top quality, high relevance
        middle_results = []  # Reinforcing content
        distributed_results = []  # Additional relevant content

        for result in results:
            result_tokens = self._estimate_result_tokens(result)
            result.importance_score = self._calculate_importance_score(result)

            # Categorize based on importance and relevance
            if result.importance_score >= 0.8 or result.relevance_score >= 0.8:
                prime_results.append(result)
            elif result.importance_score >= 0.5 or result.relevance_score >= 0.5:
                middle_results.append(result)
            else:
                distributed_results.append(result)

        # Build strategic context
        strategic_results = []
        used_tokens = 0

        # Phase 1: Fill prime positions with highest quality content
        prime_results.sort(key=lambda x: (x.importance_score, x.relevance_score), reverse=True)
        for result in prime_results:
            result_tokens = self._estimate_result_tokens(result)
            if used_tokens + result_tokens <= target_prime_tokens:
                result.placement = "prime"
                strategic_results.append(result)
                used_tokens += result_tokens

        # Phase 2: Add middle reinforcement content
        middle_results.sort(key=lambda x: (x.importance_score, x.relevance_score), reverse=True)
        target_middle_start = int(max_tokens * 0.4)  # Start at 40% mark
        for result in middle_results:
            result_tokens = self._estimate_result_tokens(result)
            if used_tokens + result_tokens <= target_middle_tokens:
                result.placement = "middle_reinforcement"
                strategic_results.append(result)
                used_tokens += result_tokens

        # Phase 3: Distribute remaining content strategically
        remaining_tokens = max_tokens - used_tokens
        distributed_slots = self._calculate_distribution_slots(
            remaining_tokens, len(distributed_results)
        )

        for i, result in enumerate(distributed_results):
            if i >= len(distributed_slots):
                break

            result_tokens = self._estimate_result_tokens(result)
            slot_tokens = distributed_slots[i]

            if result_tokens <= slot_tokens:
                result.placement = "distributed"
                strategic_results.append(result)
                used_tokens += result_tokens

        # Optimize final ordering for flow and coherence
        strategic_results = self._optimize_flow_ordering(strategic_results)

        logger.info(
            f"Strategic placement: {len(strategic_results)}/{len(results)} results "
            f"in {used_tokens}/{max_tokens} tokens "
            f"(prime: {sum(1 for r in strategic_results if r.placement == 'prime')}, "
            f"middle: {sum(1 for r in strategic_results if r.placement == 'middle_reinforcement')}, "
            f"distributed: {sum(1 for r in strategic_results if r.placement == 'distributed')})"
        )

        return strategic_results

    def _calculate_importance_score(self, result: RetrievalResult) -> float:
        """
        Calculate importance score based on multiple factors.

        Factors:
        - Recency (more recent gets higher score)
        - Relevance similarity
        - User interaction patterns
        - Content type importance
        - Cross-reference frequency
        """
        base_score = result.relevance_score

        # Recency factor (more recent = higher importance)
        days_old = (datetime.now() - result.timestamp).days
        recency_factor = max(0.1, 1.0 - (days_old / 365))  # Decay over a year

        # Content type importance
        content_importance = self._get_content_importance(result.content_type or "text")

        # User pattern importance (if available)
        pattern_importance = getattr(result, "pattern_score", 0.0)

        # Cross-reference importance (if this content is frequently referenced)
        cross_ref_importance = getattr(result, "cross_reference_count", 0) / 10.0

        # Combine factors with weights
        importance_score = (
            base_score * 0.4  # Base relevance
            + recency_factor * 0.2  # Recency
            + content_importance * 0.15  # Content type
            + pattern_importance * 0.15  # User patterns
            + cross_ref_importance * 0.1  # Cross-references
        )

        return min(1.0, importance_score)

    def _get_content_importance(self, content_type: str) -> float:
        """Get importance weight for content type."""
        importance_map = {
            "code": 0.9,  # Code snippets are highly valuable
            "error": 0.95,  # Error messages are critical
            "decision": 0.85,  # Decisions are important
            "question": 0.7,  # Questions show user intent
            "summary": 0.8,  # Summaries contain key info
            "text": 0.5,  # Default text content
        }
        return importance_map.get(content_type.lower(), 0.5)

    def _estimate_result_tokens(self, result: RetrievalResult) -> int:
        """
        More accurate token estimation for a single result.

        Accounts for metadata and formatting overhead.
        """
        # Base token count (4 chars per token on average)
        base_tokens = len(result.excerpt) // 4

        # Add overhead for metadata
        metadata_overhead = 20  # Tokens for timestamps, sources, etc.

        # Add formatting overhead
        formatting_overhead = 10  # Tokens for separators, labels

        return base_tokens + metadata_overhead + formatting_overhead

    def _calculate_distribution_slots(self, available_tokens: int, num_results: int) -> List[int]:
        """
        Calculate token slots for distributed content placement.

        Ensures even distribution throughout remaining context space.
        """
        if num_results == 0:
            return []

        # Reserve 10% buffer for safety
        usable_tokens = int(available_tokens * 0.9)

        # Calculate base slot size
        base_slot = usable_tokens // num_results

        # Distribute remaining tokens to early slots (for better UX)
        remaining_tokens = usable_tokens % num_results
        slots = []

        for i in range(num_results):
            slot_size = base_slot
            if i < remaining_tokens:
                slot_size += 1
            slots.append(slot_size)

        return slots

    def _optimize_flow_ordering(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Optimize result ordering for better flow and coherence.

        Group related items together and maintain logical progression.
        """
        if len(results) <= 2:
            return results

        # Separate by placement
        prime = [r for r in results if r.placement == "prime"]
        middle = [r for r in results if r.placement == "middle_reinforcement"]
        distributed = [r for r in results if r.placement == "distributed"]

        # Sort within each group
        prime.sort(key=lambda x: (x.importance_score, x.relevance_score), reverse=True)
        middle.sort(key=lambda x: (x.importance_score, x.relevance_score), reverse=True)

        # For distributed items, ensure variety
        distributed.sort(key=lambda x: (x.timestamp, x.importance_score))

        # Reassemble in flow order
        return prime + middle + distributed

    def _optimize_ordering(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Optimize ordering when all results fit in context.

        Even without token limits, we should optimize for information flow.
        """
        if not results:
            return results

        # Sort by importance score first, then by relevance
        optimized = sorted(
            results,
            key=lambda x: (self._calculate_importance_score(x), x.relevance_score),
            reverse=True,
        )

        # Apply flow optimization
        return self._optimize_flow_ordering(optimized)

    def _detect_quality_degradation(self, context: List[RetrievalResult]) -> Dict[str, Any]:
        """
        Detect potential quality degradation in context.

        Analyzes context for patterns that indicate information loss.
        """
        if len(context) < 3:
            return {"degradation_detected": False, "score": 1.0}

        # Calculate quality metrics
        metrics = {
            "relevance_variance": self._calculate_relevance_variance(context),
            "importance_drop": self._calculate_importance_drop(context),
            "content_distribution": self._calculate_content_distribution(context),
            "temporal_gaps": self._calculate_temporal_gaps(context),
        }

        # Overall quality score
        quality_score = (
            (1.0 - metrics["relevance_variance"]) * 0.3
            + (1.0 - metrics["importance_drop"]) * 0.3
            + metrics["content_distribution"] * 0.2
            + (1.0 - metrics["temporal_gaps"]) * 0.2
        )

        degradation_detected = quality_score < 0.7

        return {
            "degradation_detected": degradation_detected,
            "score": quality_score,
            "metrics": metrics,
            "recommendations": self._generate_quality_recommendations(metrics),
        }

    def _calculate_relevance_variance(self, context: List[RetrievalResult]) -> float:
        """Calculate variance in relevance scores (lower is better)."""
        if not context:
            return 0.0

        scores = [r.relevance_score for r in context]
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)

        # Normalize to 0-1 range
        return min(1.0, variance / 0.25)  # Max variance for scores 0-1 is 0.25

    def _calculate_importance_drop(self, context: List[RetrievalResult]) -> float:
        """Calculate drop in importance across context (lower is better)."""
        if len(context) < 2:
            return 0.0

        importance_scores = [self._calculate_importance_score(r) for r in context]

        # Calculate drop from start to end
        start_score = importance_scores[0]
        end_score = importance_scores[-1]

        if start_score == 0:
            return 1.0

        drop = (start_score - end_score) / start_score
        return max(0.0, min(1.0, drop))

    def _calculate_content_distribution(self, context: List[RetrievalResult]) -> float:
        """Calculate distribution quality of content types (higher is better)."""
        if not context:
            return 0.0

        # Count content types
        content_types = {}
        for result in context:
            content_type = result.content_type or "text"
            content_types[content_type] = content_types.get(content_type, 0) + 1

        # Good distribution has variety
        num_types = len(content_types)
        ideal_ratio = 1.0 / max(1, num_types)

        # Calculate how evenly distributed the types are
        ratios = [count / len(context) for count in content_types.values()]
        distribution_score = sum(1.0 - abs(ratio - ideal_ratio) for ratio in ratios) / len(ratios)

        # Bonus for having multiple content types
        type_bonus = min(0.3, num_types * 0.1)

        return min(1.0, distribution_score + type_bonus)

    def _calculate_temporal_gaps(self, context: List[RetrievalResult]) -> float:
        """Calculate temporal gaps in context (lower is better)."""
        if len(context) < 2:
            return 0.0

        timestamps = [r.timestamp for r in context]
        timestamps.sort()

        # Calculate gaps between consecutive items
        gaps = []
        for i in range(1, len(timestamps)):
            gap = (timestamps[i] - timestamps[i - 1]).days
            gaps.append(gap)

        # Average gap in days, normalized to 0-1 (1 year = max gap)
        avg_gap = sum(gaps) / len(gaps) if gaps else 0
        normalized_gap = min(1.0, avg_gap / 365)

        return normalized_gap

    def _generate_quality_recommendations(self, metrics: Dict[str, float]) -> List[str]:
        """Generate recommendations to improve context quality."""
        recommendations = []

        if metrics["relevance_variance"] > 0.3:
            recommendations.append(
                "High relevance variance - consider filtering low-relevance items"
            )

        if metrics["importance_drop"] > 0.4:
            recommendations.append("Significant importance drop - redistribute important items")

        if metrics["content_distribution"] < 0.5:
            recommendations.append("Poor content type distribution - include more variety")

        if metrics["temporal_gaps"] > 0.6:
            recommendations.append("Large temporal gaps - consider temporal clustering")

        return recommendations

    def _load_user_patterns(self) -> Dict[str, Any]:
        """Load user patterns from storage."""
        try:
            if not self.storage:
                return {}

            # In a full implementation, this would load from a patterns table
            # For now, extract patterns from existing conversations
            return self._extract_patterns_from_conversations()

        except Exception as e:
            logger.warning(f"Failed to load user patterns: {e}")
            return {}

    def _extract_patterns_from_conversations(self) -> Dict[str, Any]:
        """Extract patterns from existing conversations."""
        try:
            if not self.storage:
                return {}

            # Get recent conversations for pattern extraction
            recent_conversations = self.storage.get_recent_conversations(limit=50)

            if not recent_conversations:
                return {}

            # Extract patterns from these conversations
            all_results = []
            for conv in recent_conversations:
                # Create a mock result for pattern extraction
                mock_result = type(
                    "MockResult",
                    (),
                    {
                        "excerpt": conv.summary or "",
                        "relevance_score": 0.5,
                        "timestamp": conv.updated_at or conv.created_at,
                        "conversation_id": conv.id,
                    },
                )()
                all_results.append(mock_result)

            # Use existing pattern extraction
            patterns = self._extract_patterns(all_results)

            # Convert to user pattern format
            user_patterns = {}

            # Convert keywords to user patterns
            keywords = patterns.get("keywords", {}).get("importance", {})
            for keyword, importance in keywords.items():
                if importance > 0.5:
                    user_patterns[f"keyword_{keyword}"] = {
                        "type": "keyword",
                        "keyword": keyword,
                        "importance": importance,
                        "frequency": keywords.get(keyword, 0),
                        "contexts": [],
                    }

            # Convert topics to user patterns
            topics = patterns.get("topics", {}).get("scores", {})
            for topic, score in topics.items():
                if score > 0.3:
                    user_patterns[f"topic_{topic}"] = {
                        "type": "topic",
                        "topic": topic,
                        "score": score,
                        "keywords": patterns.get("topics", {}).get("results", {}).get(topic, []),
                    }

            # Convert communication style to user patterns
            styles = patterns.get("communication_style", {}).get("scores", {})
            for style, score in styles.items():
                if score > 0.3:
                    user_patterns[f"style_{style}"] = {
                        "type": "communication_style",
                        "style": style,
                        "score": score,
                        "examples": patterns.get("communication_style", {})
                        .get("examples", {})
                        .get(style, []),
                    }

            logger.info(
                f"Extracted {len(user_patterns)} user patterns from {len(recent_conversations)} conversations"
            )
            return user_patterns

        except Exception as e:
            logger.warning(f"Failed to extract patterns from conversations: {e}")
            return {}

    def _calculate_pattern_match_score(
        self, query_text: str, pattern_data: Dict[str, Any]
    ) -> float:
        """Calculate how well a query matches a stored pattern."""
        try:
            query_lower = query_text.lower()
            pattern_type = pattern_data.get("type", "")

            if pattern_type == "keyword":
                keyword = pattern_data.get("keyword", "")
                if keyword in query_lower:
                    return pattern_data.get("importance", 0.5)
                return 0.0

            elif pattern_type == "topic":
                topic_keywords = pattern_data.get("keywords", [])
                matches = sum(
                    1
                    for kw_data in topic_keywords
                    if any(kw in query_lower for kw in kw_data.get("matches", []))
                )
                return min(1.0, matches / 3.0) * pattern_data.get("score", 0.5)

            elif pattern_type == "communication_style":
                # Check if query matches communication style indicators
                style_indicators = {
                    "questioning": ["?", "how", "what", "why", "can", "could"],
                    "declarative": ["is", "are", "will", "be"],
                    "imperative": ["do", "make", "create", "implement"],
                    "expressive": ["feel", "think", "opinion"],
                }

                style = pattern_data.get("style", "")
                indicators = style_indicators.get(style, [])
                matches = sum(1 for indicator in indicators if indicator in query_lower)
                return min(1.0, matches / len(indicators)) * pattern_data.get("score", 0.5)

            return 0.0

        except Exception as e:
            logger.warning(f"Pattern match score calculation failed: {e}")
            return 0.0

    def _get_pattern_conversations(
        self, pattern_name: str, pattern_data: Dict[str, Any]
    ) -> List[Any]:
        """Get conversations associated with a specific pattern."""
        try:
            if not self.storage:
                return []

            # For now, return recent conversations as associated
            # In a full implementation, this would query a pattern_conversation table
            recent_conversations = self.storage.get_recent_conversations(limit=10)

            # Filter based on pattern relevance
            pattern_type = pattern_data.get("type", "")

            if pattern_type == "keyword":
                keyword = pattern_data.get("keyword", "")
                return [
                    conv
                    for conv in recent_conversations
                    if conv.summary and keyword.lower() in conv.summary.lower()
                ]

            elif pattern_type == "topic":
                topic_keywords = pattern_data.get("keywords", [])
                return [
                    conv
                    for conv in recent_conversations
                    if conv.summary
                    and any(
                        kw in conv.summary.lower()
                        for kw_data in topic_keywords
                        for kw in ["code", "plan", "feel", "learn", "create", "analyze"][:3]
                    )
                ]

            # Default return recent conversations
            return recent_conversations[:5]

        except Exception as e:
            logger.warning(f"Failed to get pattern conversations: {e}")
            return []

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text."""
        # Simple keyword extraction - in production, use more sophisticated NLP
        words = re.findall(r"\b\w+\b", text.lower())

        # Filter out common stop words
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "as",
            "is",
            "was",
            "are",
            "were",
            "been",
            "be",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "this",
            "that",
            "these",
            "those",
        }

        keywords = [word for word in words if len(word) > 2 and word not in stop_words]

        # Return unique keywords
        return list(set(keywords))[:10]  # Limit to 10 keywords

    def _create_keyword_excerpt(self, content: str, keywords: List[str], max_length: int) -> str:
        """Create excerpt showing keyword matches."""
        # Find first keyword occurrence
        content_lower = content.lower()
        for keyword in keywords:
            keyword_lower = keyword.lower()
            pos = content_lower.find(keyword_lower)
            if pos != -1:
                # Create excerpt around keyword
                start = max(0, pos - 50)
                end = min(len(content), pos + len(keyword) + max_length - 100)

                excerpt = content[start:end]
                if start > 0:
                    excerpt = "..." + excerpt
                if end < len(content):
                    excerpt = excerpt + "..."

                return excerpt

        # Fallback to first characters
        return content[:max_length] + ("..." if len(content) > max_length else "")

    def _estimate_tokens(self, results: List[RetrievalResult]) -> int:
        """Estimate total tokens for results."""
        # Rough estimation: 1 token ≈ 4 characters
        total_chars = sum(len(result.excerpt) for result in results)
        return total_chars // 4

    def _extract_patterns(self, results: List[RetrievalResult]) -> Dict[str, Any]:
        """Extract patterns from search results."""
        try:
            if not results:
                return {}

            # Extract various types of patterns from the results
            patterns = {
                "keywords": self._extract_keyword_patterns(results),
                "topics": self._extract_topic_patterns(results),
                "communication_style": self._extract_communication_patterns(results),
                "preferences": self._extract_preference_patterns(results),
                "temporal": self._extract_temporal_patterns(results),
                "emotional": self._extract_emotional_patterns(results),
            }

            # Calculate pattern statistics
            patterns["statistics"] = {
                "total_results": len(results),
                "pattern_density": self._calculate_pattern_density(patterns),
                "confidence": self._calculate_pattern_confidence(patterns),
            }

            logger.debug(
                f"Extracted {sum(len(v) for v in patterns.values() if isinstance(v, dict))} patterns"
            )
            return patterns

        except Exception as e:
            logger.warning(f"Pattern extraction failed: {e}")
            return {"error": str(e)}

    def _extract_keyword_patterns(self, results: List[RetrievalResult]) -> Dict[str, Any]:
        """Extract keyword usage patterns from results."""
        keyword_freq = {}
        keyword_context = {}

        for result in results:
            # Extract keywords from the result text
            keywords = self._extract_keywords(result.excerpt)

            for keyword in keywords:
                # Track frequency
                keyword_freq[keyword] = keyword_freq.get(keyword, 0) + 1

                # Track context where keywords appear
                if keyword not in keyword_context:
                    keyword_context[keyword] = []

                keyword_context[keyword].append(
                    {
                        "conversation_id": result.conversation_id,
                        "timestamp": result.timestamp,
                        "relevance": result.relevance_score,
                        "context_snippet": result.excerpt[:100] + "..."
                        if len(result.excerpt) > 100
                        else result.excerpt,
                    }
                )

        # Calculate keyword importance (frequency * relevance)
        keyword_importance = {}
        for keyword, freq in keyword_freq.items():
            contexts = keyword_context.get(keyword, [])
            avg_relevance = sum(c["relevance"] for c in contexts) / len(contexts) if contexts else 0
            keyword_importance[keyword] = freq * avg_relevance

        return {
            "frequency": keyword_freq,
            "importance": keyword_importance,
            "contexts": keyword_context,
        }

    def _extract_topic_patterns(self, results: List[RetrievalResult]) -> Dict[str, Any]:
        """Extract topic-based patterns from results."""
        topic_keywords = {
            "technical": [
                "code",
                "function",
                "class",
                "method",
                "algorithm",
                "debug",
                "error",
                "bug",
            ],
            "personal": ["feel", "think", "want", "need", "like", "dislike", "prefer"],
            "planning": [
                "plan",
                "schedule",
                "deadline",
                "goal",
                "objective",
                "strategy",
                "roadmap",
            ],
            "learning": ["learn", "understand", "explain", "clarify", "example", "tutorial"],
            "creative": ["create", "design", "imagine", "innovate", "invent", "artistic"],
            "analytical": ["analyze", "compare", "evaluate", "assess", "measure", "metric"],
        }

        topic_scores = {}
        topic_results = {}

        for topic, keywords in topic_keywords.items():
            topic_score = 0
            matching_results = []

            for result in results:
                text_lower = result.excerpt.lower()
                matches = sum(1 for keyword in keywords if keyword in text_lower)

                if matches > 0:
                    topic_score += matches * result.relevance_score
                    matching_results.append(
                        {
                            "result_id": result.id,
                            "matches": matches,
                            "relevance": result.relevance_score,
                        }
                    )

            if topic_score > 0:
                topic_scores[topic] = topic_score
                topic_results[topic] = matching_results

        return {
            "scores": topic_scores,
            "results": topic_results,
            "dominant_topic": max(topic_scores.items(), key=lambda x: x[1])[0]
            if topic_scores
            else None,
        }

    def _extract_communication_patterns(self, results: List[RetrievalResult]) -> Dict[str, Any]:
        """Extract communication style patterns from results."""
        communication_indicators = {
            "questioning": ["?", "how", "what", "why", "when", "where", "can", "could", "would"],
            "declarative": ["is", "are", "was", "were", "will", "be", "have", "has"],
            "imperative": ["do", "make", "create", "implement", "fix", "solve", "try"],
            "expressive": ["!", "feel", "think", "believe", "opinion", "view", "perspective"],
        }

        style_scores = {}
        style_examples = {}

        for style, indicators in communication_indicators.items():
            style_score = 0
            examples = []

            for result in results:
                text_lower = result.excerpt.lower()
                matches = sum(1 for indicator in indicators if indicator in text_lower)

                if matches > 0:
                    style_score += matches * result.relevance_score
                    if len(examples) < 3:  # Keep up to 3 examples
                        examples.append(
                            {
                                "text": result.excerpt[:200] + "..."
                                if len(result.excerpt) > 200
                                else result.excerpt,
                                "relevance": result.relevance_score,
                                "matches": matches,
                            }
                        )

            if style_score > 0:
                style_scores[style] = style_score
                style_examples[style] = examples

        return {
            "scores": style_scores,
            "examples": style_examples,
            "dominant_style": max(style_scores.items(), key=lambda x: x[1])[0]
            if style_scores
            else None,
        }

    def _extract_preference_patterns(self, results: List[RetrievalResult]) -> Dict[str, Any]:
        """Extract user preference patterns from results."""
        preference_indicators = {
            "prefers_detailed": ["explain", "detail", "elaborate", "more", "specific", "thorough"],
            "prefers_concise": ["brief", "short", "concise", "summary", "quick", "simple"],
            "prefers_examples": ["example", "illustrate", "demonstrate", "show", "instance"],
            "prefers_technical": [
                "technical",
                "implementation",
                "code",
                "algorithm",
                "architecture",
            ],
            "prefers_conceptual": ["concept", "theory", "principle", "idea", "approach"],
        }

        preference_scores = {}

        for preference, indicators in preference_indicators.items():
            pref_score = 0

            for result in results:
                text_lower = result.excerpt.lower()
                matches = sum(1 for indicator in indicators if indicator in text_lower)

                if matches > 0:
                    pref_score += matches * result.relevance_score

            if pref_score > 0:
                preference_scores[preference] = pref_score

        return {
            "scores": preference_scores,
            "strongest_preference": max(preference_scores.items(), key=lambda x: x[1])[0]
            if preference_scores
            else None,
        }

    def _extract_temporal_patterns(self, results: List[RetrievalResult]) -> Dict[str, Any]:
        """Extract temporal patterns from results."""
        if not results:
            return {}

        # Sort results by timestamp
        sorted_results = sorted(results, key=lambda x: x.timestamp)

        # Analyze temporal distribution
        now = datetime.now()
        time_buckets = {
            "recent": 0,  # Last 7 days
            "moderate": 0,  # 7-30 days
            "older": 0,  # 30-90 days
            "historical": 0,  # 90+ days
        }

        for result in sorted_results:
            days_old = (now - result.timestamp).days

            if days_old <= 7:
                time_buckets["recent"] += 1
            elif days_old <= 30:
                time_buckets["moderate"] += 1
            elif days_old <= 90:
                time_buckets["older"] += 1
            else:
                time_buckets["historical"] += 1

        # Calculate activity patterns
        activity_intensity = len(results) / max(1, (now - sorted_results[0].timestamp).days)

        return {
            "time_distribution": time_buckets,
            "activity_intensity": activity_intensity,
            "time_span_days": (now - sorted_results[0].timestamp).days if sorted_results else 0,
            "most_active_period": max(time_buckets.items(), key=lambda x: x[1])[0],
        }

    def _extract_emotional_patterns(self, results: List[RetrievalResult]) -> Dict[str, Any]:
        """Extract emotional tone patterns from results."""
        emotional_indicators = {
            "positive": ["good", "great", "excellent", "love", "perfect", "amazing", "wonderful"],
            "negative": ["bad", "terrible", "hate", "awful", "horrible", "frustrated", "annoyed"],
            "neutral": ["okay", "fine", "normal", "standard", "regular", "typical"],
            "curious": ["curious", "interested", "wonder", "question", "explore", "discover"],
            "frustrated": ["stuck", "confused", "difficult", "hard", "challenging", "problem"],
        }

        emotional_scores = {}

        for emotion, indicators in emotional_indicators.items():
            emotion_score = 0

            for result in results:
                text_lower = result.excerpt.lower()
                matches = sum(1 for indicator in indicators if indicator in text_lower)

                if matches > 0:
                    emotion_score += matches * result.relevance_score

            if emotion_score > 0:
                emotional_scores[emotion] = emotion_score

        return {
            "scores": emotional_scores,
            "dominant_emotion": max(emotional_scores.items(), key=lambda x: x[1])[0]
            if emotional_scores
            else None,
            "emotional_diversity": len(emotional_scores),
        }

    def _calculate_pattern_density(self, patterns: Dict[str, Any]) -> float:
        """Calculate the density of patterns found."""
        total_patterns = 0
        max_possible_patterns = 0

        for key, value in patterns.items():
            if isinstance(value, dict) and key != "statistics":
                if key in ["keywords", "topics", "communication_style", "preferences"]:
                    total_patterns += len([v for v in value.values() if v])
                    max_possible_patterns += len(value)
                elif key in ["temporal", "emotional"]:
                    total_patterns += 1 if value else 0
                    max_possible_patterns += 1

        return total_patterns / max(max_possible_patterns, 1)

    def _calculate_pattern_confidence(self, patterns: Dict[str, Any]) -> float:
        """Calculate confidence level in extracted patterns."""
        confidence_factors = []

        # Keyword diversity
        keywords = patterns.get("keywords", {}).get("frequency", {})
        if len(keywords) > 5:
            confidence_factors.append(0.8)
        elif len(keywords) > 2:
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.3)

        # Topic clarity
        topics = patterns.get("topics", {}).get("scores", {})
        if topics:
            max_score = max(topics.values())
            total_score = sum(topics.values())
            if max_score / total_score > 0.5:
                confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.5)

        # Communication style consistency
        styles = patterns.get("communication_style", {}).get("scores", {})
        if styles:
            confidence_factors.append(0.6)

        # Overall pattern density
        density = self._calculate_pattern_density(patterns)
        confidence_factors.append(density)

        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.0

    def _get_active_facets(self, query: SearchQuery) -> List[str]:
        """Get list of active search facets."""
        facets = []
        if query.include_semantic:
            facets.append("semantic")
        if query.include_keywords:
            facets.append("keywords")
        if query.include_recency:
            facets.append("recency")
        if query.include_patterns:
            facets.append("patterns")
        return facets

    def get_context_for_query(self, query_text: str, **kwargs) -> MemoryContext:
        """
        Convenience method to get context for a simple text query.

        Args:
            query_text: Text to search for
            **kwargs: Additional search parameters

        Returns:
            MemoryContext with search results
        """
        # Create SearchQuery from text
        query = SearchQuery(
            text=query_text,
            max_results=kwargs.get("max_results", self.memory_config.max_results),
            max_tokens=kwargs.get("max_tokens", 2000),
            include_semantic=kwargs.get("include_semantic", True),
            include_keywords=kwargs.get("include_keywords", True),
            include_recency=kwargs.get("include_recency", True),
            include_patterns=kwargs.get("include_patterns", True),
        )

        return self.retrieve_context(query)

    def close(self) -> None:
        """Close context retriever and cleanup resources."""
        if self.storage:
            self.storage.close()
            self.storage = None
        logger.info("ContextRetriever closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

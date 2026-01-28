"""
Personality adaptation system for dynamic learning.

This module provides time-weighted personality learning with stability controls,
enabling Mai to adapt her personality patterns based on conversation history
while maintaining core values and preventing rapid swings.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import math

from .layer_manager import PersonalityLayer, LayerType, LayerPriority
from .pattern_extractor import (
    TopicPatterns,
    SentimentPatterns,
    InteractionPatterns,
    TemporalPatterns,
    ResponseStylePatterns,
)


class AdaptationRate(Enum):
    """Personality adaptation speed settings."""

    SLOW = 0.01  # Conservative, stable changes
    MEDIUM = 0.05  # Balanced adaptation
    FAST = 0.1  # Rapid learning, less stable


@dataclass
class AdaptationConfig:
    """Configuration for personality adaptation."""

    learning_rate: AdaptationRate = AdaptationRate.MEDIUM
    max_weight_change: float = 0.1  # Maximum 10% change per update
    cooling_period_hours: int = 24  # Minimum time between major adaptations
    stability_threshold: float = 0.8  # Confidence threshold for stable changes
    enable_auto_adaptation: bool = True
    core_protection_strength: float = 1.0  # How strongly to protect core values


@dataclass
class AdaptationHistory:
    """Track adaptation history for rollback and analysis."""

    timestamp: datetime
    layer_id: str
    adaptation_type: str
    old_weight: float
    new_weight: float
    confidence: float
    reason: str


class PersonalityAdaptation:
    """
    Personality adaptation system with time-weighted learning.

    Provides controlled personality adaptation based on conversation patterns
    and user feedback while maintaining stability and protecting core values.
    """

    def __init__(self, config: Optional[AdaptationConfig] = None):
        """
        Initialize personality adaptation system.

        Args:
            config: Adaptation configuration settings
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or AdaptationConfig()
        self._adaptation_history: List[AdaptationHistory] = []
        self._last_adaptation_time: Dict[str, datetime] = {}

        # Core protection settings
        self._protected_aspects = {
            "helpfulness",
            "honesty",
            "safety",
            "respect",
            "boundaries",
        }

        # Learning state
        self._conversation_buffer: List[Dict[str, Any]] = []
        self._feedback_buffer: List[Dict[str, Any]] = []

        self.logger.info("PersonalityAdaptation initialized")

    def update_personality_layer(
        self,
        patterns: Dict[str, Any],
        layer_id: str,
        adaptation_rate: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Update a personality layer based on extracted patterns.

        Args:
            patterns: Extracted pattern data
            layer_id: Target layer identifier
            adaptation_rate: Override adaptation rate for this update

        Returns:
            Adaptation result with changes made
        """
        try:
            self.logger.info(f"Updating personality layer: {layer_id}")

            # Check cooling period
            if not self._can_adapt_layer(layer_id):
                return {
                    "status": "skipped",
                    "reason": "Cooling period active",
                    "layer_id": layer_id,
                }

            # Calculate effective adaptation rate
            effective_rate = adaptation_rate or self.config.learning_rate.value

            # Apply stability controls
            proposed_changes = self._calculate_proposed_changes(
                patterns, effective_rate
            )
            controlled_changes = self.apply_stability_controls(
                proposed_changes, layer_id
            )

            # Apply changes
            adaptation_result = self._apply_layer_changes(
                controlled_changes, layer_id, patterns
            )

            # Track adaptation
            self._track_adaptation(adaptation_result, layer_id)

            self.logger.info(f"Successfully updated layer {layer_id}")
            return adaptation_result

        except Exception as e:
            self.logger.error(f"Failed to update personality layer {layer_id}: {e}")
            return {
                "status": "error",
                "reason": str(e),
                "layer_id": layer_id,
            }

    def calculate_adaptation_rate(
        self,
        conversation_history: List[Dict[str, Any]],
        user_feedback: List[Dict[str, Any]],
    ) -> float:
        """
        Calculate optimal adaptation rate based on context.

        Args:
            conversation_history: Recent conversation data
            user_feedback: User feedback data

        Returns:
            Calculated adaptation rate
        """
        try:
            base_rate = self.config.learning_rate.value

            # Time-based adjustment
            time_weight = self._calculate_time_weight(conversation_history)

            # Feedback-based adjustment
            feedback_adjustment = self._calculate_feedback_adjustment(user_feedback)

            # Stability adjustment
            stability_adjustment = self._calculate_stability_adjustment()

            # Combine factors
            effective_rate = (
                base_rate * time_weight * feedback_adjustment * stability_adjustment
            )

            return max(0.001, min(0.2, effective_rate))

        except Exception as e:
            self.logger.error(f"Failed to calculate adaptation rate: {e}")
            return self.config.learning_rate.value

    def apply_stability_controls(
        self, proposed_changes: Dict[str, Any], current_state: str
    ) -> Dict[str, Any]:
        """
        Apply stability controls to proposed personality changes.

        Args:
            proposed_changes: Proposed personality modifications
            current_state: Current layer identifier

        Returns:
            Controlled changes respecting stability limits
        """
        try:
            controlled_changes = proposed_changes.copy()

            # Apply maximum change limits
            if "weight_change" in controlled_changes:
                max_change = self.config.max_weight_change
                proposed_change = abs(controlled_changes["weight_change"])

                if proposed_change > max_change:
                    self.logger.warning(
                        f"Limiting weight change from {proposed_change:.3f} to {max_change:.3f}"
                    )
                    # Scale down the change
                    scale_factor = max_change / proposed_change
                    controlled_changes["weight_change"] *= scale_factor

            # Apply core protection
            controlled_changes = self._apply_core_protection(controlled_changes)

            # Apply stability threshold
            if "confidence" in controlled_changes:
                if controlled_changes["confidence"] < self.config.stability_threshold:
                    self.logger.info(
                        f"Adaptation confidence {controlled_changes['confidence']:.3f} below threshold {self.config.stability_threshold}"
                    )
                    controlled_changes["status"] = "deferred"
                    controlled_changes["reason"] = "Low confidence"

            return controlled_changes

        except Exception as e:
            self.logger.error(f"Failed to apply stability controls: {e}")
            return proposed_changes

    def integrate_user_feedback(
        self, feedback_data: List[Dict[str, Any]], layer_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Integrate user feedback into layer weights.

        Args:
            feedback_data: User feedback entries
            layer_weights: Current layer weights

        Returns:
            Updated layer weights
        """
        try:
            updated_weights = layer_weights.copy()

            for feedback in feedback_data:
                layer_id = feedback.get("layer_id")
                rating = feedback.get("rating", 0)
                confidence = feedback.get("confidence", 0.5)

                if not layer_id or layer_id not in updated_weights:
                    continue

                # Calculate weight adjustment
                adjustment = self._calculate_feedback_adjustment(rating, confidence)

                # Apply adjustment with limits
                current_weight = updated_weights[layer_id]
                new_weight = current_weight + adjustment
                new_weight = max(0.0, min(1.0, new_weight))

                updated_weights[layer_id] = new_weight

                self.logger.info(
                    f"Updated layer {layer_id} weight from {current_weight:.3f} to {new_weight:.3f} based on feedback"
                )

            return updated_weights

        except Exception as e:
            self.logger.error(f"Failed to integrate user feedback: {e}")
            return layer_weights

    def import_pattern_data(
        self, pattern_extractor, conversation_range: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """
        Import and process pattern data for adaptation.

        Args:
            pattern_extractor: PatternExtractor instance
            conversation_range: Date range for pattern extraction

        Returns:
            Processed pattern data ready for adaptation
        """
        try:
            self.logger.info("Importing pattern data for adaptation")

            # Extract patterns
            raw_patterns = pattern_extractor.extract_all_patterns(conversation_range)

            # Process patterns for adaptation
            processed_patterns = {}

            # Topic patterns
            if "topic_patterns" in raw_patterns:
                topic_data = raw_patterns["topic_patterns"]
                processed_patterns["topic_adaptation"] = {
                    "interests": topic_data.get("user_interests", []),
                    "confidence": getattr(topic_data, "confidence_score", 0.5),
                    "recency_weight": self._calculate_recency_weight(topic_data),
                }

            # Sentiment patterns
            if "sentiment_patterns" in raw_patterns:
                sentiment_data = raw_patterns["sentiment_patterns"]
                processed_patterns["sentiment_adaptation"] = {
                    "emotional_tone": getattr(
                        sentiment_data, "emotional_tone", "neutral"
                    ),
                    "confidence": getattr(sentiment_data, "confidence_score", 0.5),
                    "stability_score": self._calculate_sentiment_stability(
                        sentiment_data
                    ),
                }

            # Interaction patterns
            if "interaction_patterns" in raw_patterns:
                interaction_data = raw_patterns["interaction_patterns"]
                processed_patterns["interaction_adaptation"] = {
                    "engagement_level": getattr(
                        interaction_data, "engagement_level", 0.5
                    ),
                    "response_urgency": getattr(
                        interaction_data, "response_time_avg", 0.0
                    ),
                    "confidence": getattr(interaction_data, "confidence_score", 0.5),
                }

            return processed_patterns

        except Exception as e:
            self.logger.error(f"Failed to import pattern data: {e}")
            return {}

    def export_layer_config(
        self, layer_manager, output_format: str = "json"
    ) -> Dict[str, Any]:
        """
        Export current layer configuration for backup/analysis.

        Args:
            layer_manager: LayerManager instance
            output_format: Export format (json, yaml)

        Returns:
            Layer configuration data
        """
        try:
            layers = layer_manager.list_layers()

            config_data = {
                "export_timestamp": datetime.utcnow().isoformat(),
                "total_layers": len(layers),
                "adaptation_config": {
                    "learning_rate": self.config.learning_rate.value,
                    "max_weight_change": self.config.max_weight_change,
                    "cooling_period_hours": self.config.cooling_period_hours,
                    "enable_auto_adaptation": self.config.enable_auto_adaptation,
                },
                "layers": layers,
                "adaptation_history": [
                    {
                        "timestamp": h.timestamp.isoformat(),
                        "layer_id": h.layer_id,
                        "adaptation_type": h.adaptation_type,
                        "confidence": h.confidence,
                    }
                    for h in self._adaptation_history[-20:]  # Last 20 adaptations
                ],
            }

            if output_format == "yaml":
                import yaml

                return yaml.dump(config_data, default_flow_style=False)
            else:
                return config_data

        except Exception as e:
            self.logger.error(f"Failed to export layer config: {e}")
            return {}

    def validate_layer_consistency(
        self, layers: List[PersonalityLayer], core_personality: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate layer consistency with core personality.

        Args:
            layers: List of personality layers
            core_personality: Core personality configuration

        Returns:
            Validation results
        """
        try:
            validation_results = {
                "valid": True,
                "conflicts": [],
                "warnings": [],
                "recommendations": [],
            }

            for layer in layers:
                # Check for core conflicts
                conflicts = self._check_core_conflicts(layer, core_personality)
                if conflicts:
                    validation_results["conflicts"].extend(conflicts)
                    validation_results["valid"] = False

                # Check for layer conflicts
                layer_conflicts = self._check_layer_conflicts(layer, layers)
                if layer_conflicts:
                    validation_results["warnings"].extend(layer_conflicts)

                # Check weight distribution
                if layer.weight > 0.9:
                    validation_results["warnings"].append(
                        f"Layer {layer.id} has very high weight ({layer.weight:.3f})"
                    )

            # Overall recommendations
            if validation_results["warnings"]:
                validation_results["recommendations"].append(
                    "Consider adjusting layer weights to prevent dominance"
                )

            if not validation_results["valid"]:
                validation_results["recommendations"].append(
                    "Resolve core conflicts before applying personality layers"
                )

            return validation_results

        except Exception as e:
            self.logger.error(f"Failed to validate layer consistency: {e}")
            return {"valid": False, "error": str(e)}

    def get_adaptation_history(
        self, layer_id: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get adaptation history for analysis.

        Args:
            layer_id: Optional layer filter
            limit: Maximum number of entries to return

        Returns:
            Adaptation history entries
        """
        history = self._adaptation_history

        if layer_id:
            history = [h for h in history if h.layer_id == layer_id]

        return [
            {
                "timestamp": h.timestamp.isoformat(),
                "layer_id": h.layer_id,
                "adaptation_type": h.adaptation_type,
                "old_weight": h.old_weight,
                "new_weight": h.new_weight,
                "confidence": h.confidence,
                "reason": h.reason,
            }
            for h in history[-limit:]
        ]

    # Private methods

    def _can_adapt_layer(self, layer_id: str) -> bool:
        """Check if layer can be adapted (cooling period)."""
        if layer_id not in self._last_adaptation_time:
            return True

        last_time = self._last_adaptation_time[layer_id]
        cooling_period = timedelta(hours=self.config.cooling_period_hours)

        return datetime.utcnow() - last_time >= cooling_period

    def _calculate_proposed_changes(
        self, patterns: Dict[str, Any], adaptation_rate: float
    ) -> Dict[str, Any]:
        """Calculate proposed changes based on patterns."""
        changes = {"adaptation_rate": adaptation_rate}

        # Calculate weight changes based on pattern confidence
        total_confidence = 0.0
        pattern_count = 0

        for pattern_name, pattern_data in patterns.items():
            if hasattr(pattern_data, "confidence_score"):
                total_confidence += pattern_data.confidence_score
                pattern_count += 1
            elif isinstance(pattern_data, dict) and "confidence" in pattern_data:
                total_confidence += pattern_data["confidence"]
                pattern_count += 1

        if pattern_count > 0:
            avg_confidence = total_confidence / pattern_count
            weight_change = adaptation_rate * avg_confidence
            changes["weight_change"] = weight_change
            changes["confidence"] = avg_confidence

        return changes

    def _apply_core_protection(self, changes: Dict[str, Any]) -> Dict[str, Any]:
        """Apply core value protection to changes."""
        protected_changes = changes.copy()

        # Reduce changes that might affect core values
        if "weight_change" in protected_changes:
            # Limit changes that could override core personality
            max_safe_change = self.config.max_weight_change * (
                1.0 - self.config.core_protection_strength
            )
            protected_changes["weight_change"] = min(
                protected_changes["weight_change"], max_safe_change
            )

        return protected_changes

    def _apply_layer_changes(
        self, changes: Dict[str, Any], layer_id: str, patterns: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply calculated changes to layer."""
        # This would integrate with LayerManager
        # For now, return the adaptation result
        return {
            "status": "applied",
            "layer_id": layer_id,
            "changes": changes,
            "patterns_used": list(patterns.keys()),
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _track_adaptation(self, result: Dict[str, Any], layer_id: str):
        """Track adaptation in history."""
        if result["status"] == "applied":
            history_entry = AdaptationHistory(
                timestamp=datetime.utcnow(),
                layer_id=layer_id,
                adaptation_type=result.get("adaptation_type", "automatic"),
                old_weight=result.get("old_weight", 0.0),
                new_weight=result.get("new_weight", 0.0),
                confidence=result.get("confidence", 0.0),
                reason=result.get("reason", "Pattern-based adaptation"),
            )

            self._adaptation_history.append(history_entry)
            self._last_adaptation_time[layer_id] = datetime.utcnow()

    def _calculate_time_weight(
        self, conversation_history: List[Dict[str, Any]]
    ) -> float:
        """Calculate time-based weight for adaptation."""
        if not conversation_history:
            return 0.5

        # Recent conversations have more weight
        now = datetime.utcnow()
        total_weight = 0.0
        total_conversations = len(conversation_history)

        for conv in conversation_history:
            conv_time = conv.get("timestamp", now)
            if isinstance(conv_time, str):
                conv_time = datetime.fromisoformat(conv_time)

            hours_ago = (now - conv_time).total_seconds() / 3600
            time_weight = math.exp(-hours_ago / 24)  # 24-hour half-life
            total_weight += time_weight

        return total_weight / total_conversations if total_conversations > 0 else 0.5

    def _calculate_feedback_adjustment(
        self, user_feedback: List[Dict[str, Any]]
    ) -> float:
        """Calculate adjustment factor based on user feedback."""
        if not user_feedback:
            return 1.0

        positive_feedback = sum(1 for fb in user_feedback if fb.get("rating", 0) > 0.5)
        total_feedback = len(user_feedback)

        if total_feedback == 0:
            return 1.0

        feedback_ratio = positive_feedback / total_feedback
        return 0.5 + feedback_ratio  # Range: 0.5 to 1.5

    def _calculate_stability_adjustment(self) -> float:
        """Calculate adjustment based on recent stability."""
        recent_history = [
            h
            for h in self._adaptation_history[-10:]
            if (datetime.utcnow() - h.timestamp).total_seconds()
            < 86400 * 7  # Last 7 days
        ]

        if len(recent_history) < 3:
            return 1.0

        # Check for volatility
        weight_changes = [abs(h.new_weight - h.old_weight) for h in recent_history]
        avg_change = sum(weight_changes) / len(weight_changes)

        # Reduce adaptation if too volatile
        if avg_change > 0.2:  # High volatility
            return 0.5
        elif avg_change > 0.1:  # Medium volatility
            return 0.8
        else:
            return 1.0

    def _calculate_feedback_adjustment(self, rating: float, confidence: float) -> float:
        """Calculate weight adjustment from feedback."""
        # Normalize rating to -1 to 1 range
        normalized_rating = (rating - 0.5) * 2

        # Apply confidence weighting
        adjustment = normalized_rating * confidence * 0.1  # Max 10% change

        return adjustment

    def _calculate_recency_weight(self, pattern_data: Any) -> float:
        """Calculate recency weight for pattern data."""
        # This would integrate with actual pattern timestamps
        return 0.8  # Placeholder

    def _calculate_sentiment_stability(self, sentiment_data: Any) -> float:
        """Calculate stability score for sentiment patterns."""
        # This would analyze sentiment consistency over time
        return 0.7  # Placeholder

    def _check_core_conflicts(
        self, layer: PersonalityLayer, core_personality: Dict[str, Any]
    ) -> List[str]:
        """Check for conflicts with core personality."""
        conflicts = []

        for modification in layer.system_prompt_modifications:
            for protected_aspect in self._protected_aspects:
                if f"not {protected_aspect}" in modification.lower():
                    conflicts.append(
                        f"Layer {layer.id} conflicts with core value: {protected_aspect}"
                    )

        return conflicts

    def _check_layer_conflicts(
        self, layer: PersonalityLayer, all_layers: List[PersonalityLayer]
    ) -> List[str]:
        """Check for conflicts with other layers."""
        conflicts = []

        for other_layer in all_layers:
            if other_layer.id == layer.id:
                continue

            # Check for contradictory modifications
            for mod1 in layer.system_prompt_modifications:
                for mod2 in other_layer.system_prompt_modifications:
                    if self._are_contradictory(mod1, mod2):
                        conflicts.append(
                            f"Layer {layer.id} contradicts layer {other_layer.id}"
                        )

        return conflicts

    def _are_contradictory(self, mod1: str, mod2: str) -> bool:
        """Check if two modifications are contradictory."""
        # Simple contradiction detection
        opposite_pairs = [
            ("formal", "casual"),
            ("verbose", "concise"),
            ("humorous", "serious"),
            ("enthusiastic", "reserved"),
        ]

        mod1_lower = mod1.lower()
        mod2_lower = mod2.lower()

        for pair in opposite_pairs:
            if pair[0] in mod1_lower and pair[1] in mod2_lower:
                return True
            if pair[1] in mod1_lower and pair[0] in mod2_lower:
                return True

        return False

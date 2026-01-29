"""
Personality layer management system.

This module manages personality layers created from extracted patterns,
including layer creation, conflict resolution, activation, and application.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

from .pattern_extractor import (
    TopicPatterns,
    SentimentPatterns,
    InteractionPatterns,
    TemporalPatterns,
    ResponseStylePatterns,
)


class LayerType(Enum):
    """Types of personality layers."""

    TOPIC_BASED = "topic_based"
    SENTIMENT_BASED = "sentiment_based"
    INTERACTION_BASED = "interaction_based"
    TEMPORAL_BASED = "temporal_based"
    STYLE_BASED = "style_based"


class LayerPriority(Enum):
    """Priority levels for layer application."""

    CORE = 0  # Core personality values (cannot be overridden)
    HIGH = 1  # Important learned patterns
    MEDIUM = 2  # Moderate learned patterns
    LOW = 3  # Minor learned patterns


@dataclass
class PersonalityLayer:
    """
    Individual personality layer with application rules.

    Represents a learned personality pattern that can be applied
    as an overlay to the core personality.
    """

    id: str
    name: str
    layer_type: LayerType
    priority: LayerPriority
    weight: float = 1.0  # Influence strength (0.0-1.0)
    confidence: float = 0.0  # Pattern extraction confidence
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)

    # Layer content
    system_prompt_modifications: List[str] = field(default_factory=list)
    behavior_adjustments: Dict[str, Any] = field(default_factory=dict)
    response_style_changes: Dict[str, Any] = field(default_factory=dict)

    # Application rules
    activation_conditions: Dict[str, Any] = field(default_factory=dict)
    context_requirements: List[str] = field(default_factory=list)
    conflict_resolution: str = "merge"  # merge, override, skip

    # Stability tracking
    application_count: int = 0
    success_rate: float = 0.0
    user_feedback: List[Dict[str, Any]] = field(default_factory=list)

    def is_active(self, context: Dict[str, Any]) -> bool:
        """
        Check if this layer should be active in the given context.

        Args:
            context: Current conversation context

        Returns:
            True if layer should be active
        """
        # Check activation conditions
        for condition, value in self.activation_conditions.items():
            if condition in context:
                if isinstance(value, (list, set)):
                    if context[condition] not in value:
                        return False
                elif context[condition] != value:
                    return False

        # Check context requirements
        if self.context_requirements:
            context_topics = context.get("topics", [])
            if not any(req in context_topics for req in self.context_requirements):
                return False

        return True

    def calculate_effective_weight(self, context: Dict[str, Any]) -> float:
        """
        Calculate effective weight based on context and layer properties.

        Args:
            context: Current conversation context

        Returns:
            Effective weight (0.0-1.0)
        """
        base_weight = self.weight

        # Adjust based on confidence
        confidence_adjustment = self.confidence

        # Adjust based on success rate
        success_adjustment = self.success_rate

        # Adjust based on recency (more recent layers have slightly higher weight)
        days_since_creation = (datetime.utcnow() - self.created_at).days
        recency_adjustment = max(0.0, 1.0 - (days_since_creation / 365.0))

        # Combine adjustments
        effective_weight = base_weight * (
            0.4
            + 0.3 * confidence_adjustment
            + 0.2 * success_adjustment
            + 0.1 * recency_adjustment
        )

        return min(1.0, max(0.0, effective_weight))


class LayerManager:
    """
    Personality layer management system.

    Manages creation, storage, activation, and application of personality
    layers with conflict resolution and priority handling.
    """

    def __init__(self):
        """Initialize layer manager."""
        self.logger = logging.getLogger(__name__)
        self._layers: Dict[str, PersonalityLayer] = {}
        self._active_layers: Set[str] = set()
        self._layer_history: List[Dict[str, Any]] = []

        # Core personality protection
        self._protected_core_values = [
            "helpfulness",
            "honesty",
            "safety",
            "respect",
            "boundaries",
        ]

    def create_layer_from_patterns(
        self,
        layer_id: str,
        layer_name: str,
        patterns: Dict[str, Any],
        priority: LayerPriority = LayerPriority.MEDIUM,
        weight: float = 1.0,
    ) -> PersonalityLayer:
        """
        Create a personality layer from extracted patterns.

        Args:
            layer_id: Unique layer identifier
            layer_name: Human-readable layer name
            patterns: Extracted pattern data
            priority: Layer priority for conflict resolution
            weight: Base influence weight

        Returns:
            Created PersonalityLayer
        """
        try:
            self.logger.info(f"Creating personality layer: {layer_name}")

            # Determine layer type from patterns
            layer_type = self._determine_layer_type(patterns)

            # Extract layer content from patterns
            system_prompt_mods = self._extract_system_prompt_modifications(patterns)
            behavior_adjustments = self._extract_behavior_adjustments(patterns)
            style_changes = self._extract_style_changes(patterns)

            # Set activation conditions based on pattern type
            activation_conditions = self._determine_activation_conditions(patterns)

            # Calculate confidence from pattern data
            confidence = self._calculate_layer_confidence(patterns)

            # Create the layer
            layer = PersonalityLayer(
                id=layer_id,
                name=layer_name,
                layer_type=layer_type,
                priority=priority,
                weight=weight,
                confidence=confidence,
                system_prompt_modifications=system_prompt_mods,
                behavior_adjustments=behavior_adjustments,
                response_style_changes=style_changes,
                activation_conditions=activation_conditions,
            )

            # Store the layer
            self._layers[layer_id] = layer

            # Log layer creation
            self._layer_history.append(
                {
                    "action": "created",
                    "layer_id": layer_id,
                    "layer_name": layer_name,
                    "timestamp": datetime.utcnow().isoformat(),
                    "patterns": patterns,
                }
            )

            self.logger.info(f"Successfully created personality layer: {layer_name}")
            return layer

        except Exception as e:
            self.logger.error(f"Failed to create personality layer {layer_name}: {e}")
            raise

    def get_active_layers(self, context: Dict[str, Any]) -> List[PersonalityLayer]:
        """
        Get all active layers for the given context.

        Args:
            context: Current conversation context

        Returns:
            List of active layers sorted by priority and weight
        """
        try:
            active_layers = []

            for layer in self._layers.values():
                if layer.is_active(context):
                    # Calculate effective weight for this context
                    effective_weight = layer.calculate_effective_weight(context)

                    # Only include layers with meaningful weight
                    if effective_weight > 0.1:
                        active_layers.append((layer, effective_weight))

            # Sort by priority first, then by effective weight
            active_layers.sort(key=lambda x: (x[0].priority.value, -x[1]))

            # Return just the layers (not the weights)
            return [layer for layer, _ in active_layers]

        except Exception as e:
            self.logger.error(f"Failed to get active layers: {e}")
            return []

    def apply_layers(
        self, base_system_prompt: str, context: Dict[str, Any], max_layers: int = 5
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Apply active personality layers to system prompt and behavior.

        Args:
            base_system_prompt: Original system prompt
            context: Current conversation context
            max_layers: Maximum number of layers to apply

        Returns:
            Tuple of (modified_system_prompt, behavior_adjustments)
        """
        try:
            self.logger.info("Applying personality layers")

            # Get active layers
            active_layers = self.get_active_layers(context)[:max_layers]

            if not active_layers:
                return base_system_prompt, {}

            # Start with base prompt
            modified_prompt = base_system_prompt
            behavior_adjustments = {}
            style_adjustments = {}

            # Apply layers in priority order
            for layer in active_layers:
                # Check for conflicts with core values
                if not self._is_core_safe(layer):
                    self.logger.warning(
                        f"Skipping layer {layer.id} - conflicts with core values"
                    )
                    continue

                # Apply system prompt modifications
                for modification in layer.system_prompt_modifications:
                    modified_prompt = self._apply_prompt_modification(
                        modified_prompt, modification, layer.confidence
                    )

                # Apply behavior adjustments
                behavior_adjustments.update(layer.behavior_adjustments)
                style_adjustments.update(layer.response_style_changes)

                # Track application
                layer.application_count += 1
                layer.last_updated = datetime.utcnow()

            # Combine style adjustments into behavior
            behavior_adjustments.update(style_adjustments)

            self.logger.info(f"Applied {len(active_layers)} personality layers")
            return modified_prompt, behavior_adjustments

        except Exception as e:
            self.logger.error(f"Failed to apply personality layers: {e}")
            return base_system_prompt, {}

    def update_layer_feedback(self, layer_id: str, feedback: Dict[str, Any]) -> bool:
        """
        Update layer with user feedback.

        Args:
            layer_id: Layer identifier
            feedback: Feedback data including rating and comments

        Returns:
            True if update successful
        """
        try:
            if layer_id not in self._layers:
                self.logger.error(f"Layer {layer_id} not found for feedback update")
                return False

            layer = self._layers[layer_id]

            # Add feedback
            feedback_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "rating": feedback.get("rating", 0),
                "comment": feedback.get("comment", ""),
                "context": feedback.get("context", {}),
            }
            layer.user_feedback.append(feedback_entry)

            # Update success rate based on feedback
            self._update_success_rate(layer)

            # Log feedback
            self._layer_history.append(
                {
                    "action": "feedback",
                    "layer_id": layer_id,
                    "feedback": feedback_entry,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

            self.logger.info(f"Updated feedback for layer {layer_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to update layer feedback: {e}")
            return False

    def get_layer_info(self, layer_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a layer.

        Args:
            layer_id: Layer identifier

        Returns:
            Layer information dictionary or None if not found
        """
        if layer_id not in self._layers:
            return None

        layer = self._layers[layer_id]
        return {
            "id": layer.id,
            "name": layer.name,
            "type": layer.layer_type.value,
            "priority": layer.priority.value,
            "weight": layer.weight,
            "confidence": layer.confidence,
            "created_at": layer.created_at.isoformat(),
            "last_updated": layer.last_updated.isoformat(),
            "application_count": layer.application_count,
            "success_rate": layer.success_rate,
            "activation_conditions": layer.activation_conditions,
            "user_feedback_count": len(layer.user_feedback),
        }

    def list_layers(
        self, layer_type: Optional[LayerType] = None
    ) -> List[Dict[str, Any]]:
        """
        List all layers, optionally filtered by type.

        Args:
            layer_type: Optional layer type filter

        Returns:
            List of layer information dictionaries
        """
        layers = []

        for layer in self._layers.values():
            if layer_type and layer.layer_type != layer_type:
                continue

            layers.append(self.get_layer_info(layer.id))

        return sorted(layers, key=lambda x: (x["priority"], -x["weight"]))

    def delete_layer(self, layer_id: str) -> bool:
        """
        Delete a personality layer.

        Args:
            layer_id: Layer identifier

        Returns:
            True if deletion successful
        """
        try:
            if layer_id not in self._layers:
                return False

            # Remove from storage
            del self._layers[layer_id]

            # Remove from active set if present
            self._active_layers.discard(layer_id)

            # Log deletion
            self._layer_history.append(
                {
                    "action": "deleted",
                    "layer_id": layer_id,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

            self.logger.info(f"Deleted personality layer: {layer_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete layer {layer_id}: {e}")
            return False

    def _determine_layer_type(self, patterns: Dict[str, Any]) -> LayerType:
        """Determine layer type from pattern data."""
        if "topic_patterns" in patterns:
            return LayerType.TOPIC_BASED
        elif "sentiment_patterns" in patterns:
            return LayerType.SENTIMENT_BASED
        elif "interaction_patterns" in patterns:
            return LayerType.INTERACTION_BASED
        elif "temporal_patterns" in patterns:
            return LayerType.TEMPORAL_BASED
        elif "response_style_patterns" in patterns:
            return LayerType.STYLE_BASED
        else:
            return LayerType.MEDIUM  # Default

    def _extract_system_prompt_modifications(
        self, patterns: Dict[str, Any]
    ) -> List[str]:
        """Extract system prompt modifications from patterns."""
        modifications = []

        # Topic-based modifications
        if "topic_patterns" in patterns:
            topic_patterns = patterns["topic_patterns"]
            if topic_patterns.user_interests:
                interests = ", ".join(topic_patterns.user_interests[:3])
                modifications.append(f"Show interest and knowledge about: {interests}")

        # Sentiment-based modifications
        if "sentiment_patterns" in patterns:
            sentiment_patterns = patterns["sentiment_patterns"]
            if sentiment_patterns.emotional_tone == "positive":
                modifications.append("Maintain a positive and encouraging tone")
            elif sentiment_patterns.emotional_tone == "negative":
                modifications.append("Be more empathetic and understanding")

        # Interaction-based modifications
        if "interaction_patterns" in patterns:
            interaction_patterns = patterns["interaction_patterns"]
            if interaction_patterns.question_frequency > 0.5:
                modifications.append(
                    "Ask clarifying questions to understand needs better"
                )
            if interaction_patterns.engagement_level > 0.7:
                modifications.append("Show enthusiasm and engagement in conversations")

        # Style-based modifications
        if "response_style_patterns" in patterns:
            style_patterns = patterns["response_style_patterns"]
            if style_patterns.formality_level > 0.7:
                modifications.append("Use more formal and professional language")
            elif style_patterns.formality_level < 0.3:
                modifications.append("Use casual and friendly language")
            if style_patterns.humor_frequency > 0.3:
                modifications.append("Include appropriate humor and wit")

        return modifications

    def _extract_behavior_adjustments(self, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Extract behavior adjustments from patterns."""
        adjustments = {}

        # Response time adjustments
        if "interaction_patterns" in patterns:
            interaction = patterns["interaction_patterns"]
            if interaction.response_time_avg > 0:
                adjustments["response_urgency"] = min(
                    1.0, interaction.response_time_avg / 60.0
                )

        # Conversation balance
        if "interaction_patterns" in patterns:
            interaction = patterns["interaction_patterns"]
            if interaction.conversation_balance > 0.7:
                adjustments["talkativeness"] = "low"
            elif interaction.conversation_balance < 0.3:
                adjustments["talkativeness"] = "high"

        return adjustments

    def _extract_style_changes(self, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Extract response style changes from patterns."""
        style_changes = {}

        if "response_style_patterns" in patterns:
            style = patterns["response_style_patterns"]
            style_changes["formality"] = style.formality_level
            style_changes["verbosity"] = style.verbosity
            style_changes["emoji_usage"] = style.emoji_usage
            style_changes["humor_level"] = style.humor_frequency
            style_changes["directness"] = style.directness

        return style_changes

    def _determine_activation_conditions(
        self, patterns: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Determine activation conditions from patterns."""
        conditions = {}

        # Topic-based activation
        if "topic_patterns" in patterns:
            topic_patterns = patterns["topic_patterns"]
            if topic_patterns.user_interests:
                conditions["topics"] = topic_patterns.user_interests

        # Temporal-based activation
        if "temporal_patterns" in patterns:
            temporal = patterns["temporal_patterns"]
            if temporal.preferred_times:
                preferred_hours = [
                    int(hour) for hour, _ in temporal.preferred_times[:3]
                ]
                conditions["hour"] = preferred_hours

        return conditions

    def _calculate_layer_confidence(self, patterns: Dict[str, Any]) -> float:
        """Calculate overall layer confidence from pattern confidences."""
        confidences = []

        for pattern_name, pattern_data in patterns.items():
            if hasattr(pattern_data, "confidence_score"):
                confidences.append(pattern_data.confidence_score)
            elif isinstance(pattern_data, dict) and "confidence_score" in pattern_data:
                confidences.append(pattern_data["confidence_score"])

        if confidences:
            return sum(confidences) / len(confidences)
        else:
            return 0.5  # Default confidence

    def _is_core_safe(self, layer: PersonalityLayer) -> bool:
        """Check if layer conflicts with core personality values."""
        # Check system prompt modifications for conflicts
        for modification in layer.system_prompt_modifications:
            modification_lower = modification.lower()

            # Check for conflicts with protected values
            for protected_value in self._protected_core_values:
                if f"not {protected_value}" in modification_lower:
                    return False
                if f"avoid {protected_value}" in modification_lower:
                    return False

        return True

    def _apply_prompt_modification(
        self, base_prompt: str, modification: str, confidence: float
    ) -> str:
        """Apply a modification to the system prompt."""
        # Simple concatenation with confidence-based wording
        if confidence > 0.8:
            return f"{base_prompt}\n\n{modification}"
        elif confidence > 0.5:
            return f"{base_prompt}\n\nConsider: {modification}"
        else:
            return f"{base_prompt}\n\nOptionally: {modification}"

    def _update_success_rate(self, layer: PersonalityLayer) -> None:
        """Update layer success rate based on feedback."""
        if not layer.user_feedback:
            layer.success_rate = 0.5  # Default
            return

        # Calculate average rating from feedback
        ratings = [fb["rating"] for fb in layer.user_feedback if "rating" in fb]
        if ratings:
            layer.success_rate = sum(ratings) / len(ratings)
        else:
            layer.success_rate = 0.5

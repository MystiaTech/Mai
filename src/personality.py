"""
Mai's personality system with memory learning integration.

This module provides the main personality interface that combines core personality
values with learned personality layers from the memory system. It maintains
Mai's essential character while allowing adaptive learning from conversations.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Import core personality from resource system
try:
    from src.resource.personality import get_core_personality, get_personality_response
except ImportError:
    # Fallback if resource system not available
    def get_core_personality():
        return {
            "name": "Mai",
            "core_values": ["helpful", "honest", "safe", "respectful", "boundaries"],
            "communication_style": "warm and professional",
            "response_patterns": ["clarifying", "supportive", "informative"],
        }

    def get_personality_response(context, user_input):
        return "I'm Mai, here to help you."


# Import memory learning components
try:
    from src.memory import PersonalityLearner

    MEMORY_LEARNING_AVAILABLE = True
except ImportError:
    MEMORY_LEARNING_AVAILABLE = False
    PersonalityLearner = None


class PersonalitySystem:
    """
    Main personality system that combines core values with learned adaptations.

    Maintains Mai's essential character while integrating learned personality
    layers from conversation patterns and user feedback.
    """

    def __init__(self, memory_manager=None, enable_learning: bool = True):
        """
        Initialize personality system.

        Args:
            memory_manager: Optional MemoryManager for learning integration
            enable_learning: Whether to enable personality learning
        """
        self.logger = logging.getLogger(__name__)
        self.enable_learning = enable_learning and MEMORY_LEARNING_AVAILABLE
        self.memory_manager = memory_manager
        self.personality_learner = None

        # Load core personality
        self.core_personality = get_core_personality()
        self.protected_values = set(self.core_personality.get("core_values", []))

        # Initialize learning if available
        if self.enable_learning and memory_manager:
            try:
                self.personality_learner = memory_manager.personality_learner
                self.logger.info("Personality learning system initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize personality learning: {e}")
                self.enable_learning = False

        self.logger.info("PersonalitySystem initialized")

    def get_personality_response(
        self, context: Dict[str, Any], user_input: str, apply_learning: bool = True
    ) -> Dict[str, Any]:
        """
        Generate personality-enhanced response.

        Args:
            context: Current conversation context
            user_input: User's input message
            apply_learning: Whether to apply learned personality layers

        Returns:
            Enhanced response with personality applied
        """
        try:
            # Start with core personality response
            base_response = get_personality_response(context, user_input)

            if not apply_learning or not self.enable_learning:
                return {
                    "response": base_response,
                    "personality_applied": "core_only",
                    "active_layers": [],
                    "modifications": {},
                }

            # Apply learned personality layers
            learning_result = self.personality_learner.apply_learning(context)

            if learning_result["status"] == "applied":
                # Enhance response with learned personality
                enhanced_response = self._apply_learned_enhancements(
                    base_response, learning_result
                )

                return {
                    "response": enhanced_response,
                    "personality_applied": "core_plus_learning",
                    "active_layers": learning_result["active_layers"],
                    "modifications": learning_result["behavior_adjustments"],
                    "layer_count": learning_result["layer_count"],
                }
            else:
                return {
                    "response": base_response,
                    "personality_applied": "core_only",
                    "active_layers": [],
                    "modifications": {},
                    "learning_status": learning_result["status"],
                }

        except Exception as e:
            self.logger.error(f"Failed to generate personality response: {e}")
            return {
                "response": get_personality_response(context, user_input),
                "personality_applied": "fallback",
                "error": str(e),
            }

    def apply_personality_layers(
        self, base_response: str, context: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Apply personality layers to a base response.

        Args:
            base_response: Original response text
            context: Current conversation context

        Returns:
            Tuple of (enhanced_response, applied_modifications)
        """
        if not self.enable_learning or not self.personality_learner:
            return base_response, {}

        try:
            learning_result = self.personality_learner.apply_learning(context)

            if learning_result["status"] == "applied":
                enhanced_response = self._apply_learned_enhancements(
                    base_response, learning_result
                )
                return enhanced_response, learning_result["behavior_adjustments"]
            else:
                return base_response, {}

        except Exception as e:
            self.logger.error(f"Failed to apply personality layers: {e}")
            return base_response, {}

    def get_active_layers(
        self, conversation_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Get currently active personality layers.

        Args:
            conversation_context: Current conversation context

        Returns:
            List of active personality layer information
        """
        if not self.enable_learning or not self.personality_learner:
            return []

        try:
            current_personality = self.personality_learner.get_current_personality()
            return current_personality.get("layers", [])
        except Exception as e:
            self.logger.error(f"Failed to get active layers: {e}")
            return []

    def validate_personality_consistency(
        self, applied_layers: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate that applied layers don't conflict with core personality.

        Args:
            applied_layers: List of applied personality layers

        Returns:
            Validation results
        """
        try:
            validation_result = {
                "valid": True,
                "conflicts": [],
                "warnings": [],
                "core_protection_active": True,
            }

            # Check each layer for core conflicts
            for layer in applied_layers:
                layer_modifications = layer.get("system_prompt_modifications", [])

                for modification in layer_modifications:
                    # Check for conflicts with protected values
                    modification_lower = modification.lower()

                    for protected_value in self.protected_values:
                        if f"not {protected_value}" in modification_lower:
                            validation_result["conflicts"].append(
                                {
                                    "layer_id": layer.get("id"),
                                    "protected_value": protected_value,
                                    "conflicting_modification": modification,
                                }
                            )
                            validation_result["valid"] = False

                        if f"avoid {protected_value}" in modification_lower:
                            validation_result["warnings"].append(
                                {
                                    "layer_id": layer.get("id"),
                                    "protected_value": protected_value,
                                    "warning_modification": modification,
                                }
                            )

            return validation_result

        except Exception as e:
            self.logger.error(f"Failed to validate personality consistency: {e}")
            return {"valid": False, "error": str(e)}

    def update_personality_feedback(
        self, layer_id: str, feedback: Dict[str, Any]
    ) -> bool:
        """
        Update personality layer with user feedback.

        Args:
            layer_id: Layer identifier
            feedback: Feedback data including rating and comments

        Returns:
            True if update successful
        """
        if not self.enable_learning or not self.personality_learner:
            return False

        try:
            return self.personality_learner.update_feedback(layer_id, feedback)
        except Exception as e:
            self.logger.error(f"Failed to update personality feedback: {e}")
            return False

    def get_personality_state(self) -> Dict[str, Any]:
        """
        Get current personality system state.

        Returns:
            Comprehensive personality state information
        """
        try:
            state = {
                "core_personality": self.core_personality,
                "protected_values": list(self.protected_values),
                "learning_enabled": self.enable_learning,
                "memory_integration": self.memory_manager is not None,
                "timestamp": datetime.utcnow().isoformat(),
            }

            if self.enable_learning and self.personality_learner:
                current_personality = self.personality_learner.get_current_personality()
                state.update(
                    {
                        "total_layers": current_personality.get("total_layers", 0),
                        "active_layers": current_personality.get("active_layers", 0),
                        "layer_types": current_personality.get("layer_types", []),
                        "recent_adaptations": current_personality.get(
                            "recent_adaptations", 0
                        ),
                        "adaptation_enabled": current_personality.get(
                            "adaptation_enabled", False
                        ),
                        "learning_rate": current_personality.get(
                            "learning_rate", "medium"
                        ),
                    }
                )

            return state

        except Exception as e:
            self.logger.error(f"Failed to get personality state: {e}")
            return {"error": str(e), "core_personality": self.core_personality}

    def trigger_learning_cycle(
        self, conversation_range: Optional[Tuple[datetime, datetime]] = None
    ) -> Dict[str, Any]:
        """
        Trigger a personality learning cycle.

        Args:
            conversation_range: Optional date range for learning

        Returns:
            Learning cycle results
        """
        if not self.enable_learning or not self.personality_learner:
            return {"status": "disabled", "message": "Personality learning not enabled"}

        try:
            if not conversation_range:
                # Default to last 30 days
                from datetime import timedelta

                end_date = datetime.utcnow()
                start_date = end_date - timedelta(days=30)
                conversation_range = (start_date, end_date)

            learning_result = self.personality_learner.learn_from_conversations(
                conversation_range
            )

            self.logger.info(
                f"Personality learning cycle completed: {learning_result.get('status')}"
            )

            return learning_result

        except Exception as e:
            self.logger.error(f"Failed to trigger learning cycle: {e}")
            return {"status": "error", "error": str(e)}

    def _apply_learned_enhancements(
        self, base_response: str, learning_result: Dict[str, Any]
    ) -> str:
        """
        Apply learned personality enhancements to base response.

        Args:
            base_response: Original response
            learning_result: Learning system results

        Returns:
            Enhanced response
        """
        try:
            enhanced_response = base_response
            behavior_adjustments = learning_result.get("behavior_adjustments", {})

            # Apply behavior adjustments
            if "talkativeness" in behavior_adjustments:
                if behavior_adjustments["talkativeness"] == "high":
                    # Add more detail and explanation
                    enhanced_response += "\n\nIs there anything specific about this you'd like me to elaborate on?"
                elif behavior_adjustments["talkativeness"] == "low":
                    # Make response more concise
                    enhanced_response = enhanced_response.split(".")[0] + "."

            if "response_urgency" in behavior_adjustments:
                urgency = behavior_adjustments["response_urgency"]
                if urgency > 0.7:
                    enhanced_response = (
                        "I'll help you right away with that. " + enhanced_response
                    )
                elif urgency < 0.3:
                    enhanced_response = (
                        "Take your time, but here's what I can help with: "
                        + enhanced_response
                    )

            # Apply style modifications from modified prompt
            modified_prompt = learning_result.get("modified_prompt", "")
            if (
                "humor" in modified_prompt.lower()
                and "formal" not in modified_prompt.lower()
            ):
                # Add light humor if appropriate
                enhanced_response = enhanced_response + " 😊"

            return enhanced_response

        except Exception as e:
            self.logger.error(f"Failed to apply learned enhancements: {e}")
            return base_response


# Global personality system instance
_personality_system: Optional[PersonalitySystem] = None


def initialize_personality(
    memory_manager=None, enable_learning: bool = True
) -> PersonalitySystem:
    """
    Initialize the global personality system.

    Args:
        memory_manager: Optional MemoryManager for learning
        enable_learning: Whether to enable personality learning

    Returns:
        Initialized PersonalitySystem instance
    """
    global _personality_system
    _personality_system = PersonalitySystem(memory_manager, enable_learning)
    return _personality_system


def get_personality_system() -> Optional[PersonalitySystem]:
    """
    Get the global personality system instance.

    Returns:
        PersonalitySystem instance or None if not initialized
    """
    return _personality_system


def get_personality_response(
    context: Dict[str, Any], user_input: str, apply_learning: bool = True
) -> Dict[str, Any]:
    """
    Get personality-enhanced response using global system.

    Args:
        context: Current conversation context
        user_input: User's input message
        apply_learning: Whether to apply learned personality layers

    Returns:
        Enhanced response with personality applied
    """
    if _personality_system:
        return _personality_system.get_personality_response(
            context, user_input, apply_learning
        )
    else:
        # Fallback to core personality only
        return {
            "response": get_personality_response(context, user_input),
            "personality_applied": "core_only",
            "active_layers": [],
            "modifications": {},
        }


def apply_personality_layers(
    base_response: str, context: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    """
    Apply personality layers using global system.

    Args:
        base_response: Original response text
        context: Current conversation context

    Returns:
        Tuple of (enhanced_response, applied_modifications)
    """
    if _personality_system:
        return _personality_system.apply_personality_layers(base_response, context)
    else:
        return base_response, {}


# Export main functions
__all__ = [
    "PersonalitySystem",
    "initialize_personality",
    "get_personality_system",
    "get_personality_response",
    "apply_personality_layers",
]

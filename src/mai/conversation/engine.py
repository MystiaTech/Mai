"""
Core Conversation Engine for Mai

This module provides the main conversation engine that orchestrates
multi-turn conversations with memory integration and natural timing.
"""

import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Tuple
from threading import Thread, Event
from dataclasses import dataclass

from ..core.interface import MaiInterface
from ..memory.manager import MemoryManager
from ..models.conversation import Conversation as ModelConversation, Message
from .state import ConversationState, ConversationTurn
from .timing import TimingCalculator
from .reasoning import ReasoningEngine
from .decomposition import RequestDecomposer
from .interruption import InterruptHandler, TurnType


logger = logging.getLogger(__name__)


@dataclass
class ConversationResponse:
    """Response from conversation processing with metadata."""

    response: str
    model_used: str
    tokens_used: int
    response_time: float
    memory_context_used: int
    timing_category: str
    conversation_id: str
    interruption_handled: bool = False
    memory_integrated: bool = False


class ConversationEngine:
    """
    Main conversation engine orchestrating multi-turn conversations.

    Integrates memory context retrieval, natural timing calculation,
    reasoning transparency, request decomposition, interruption handling,
    personality consistency, and conversation state management.
    """

    def __init__(
        self,
        mai_interface: Optional[MaiInterface] = None,
        memory_manager: Optional[MemoryManager] = None,
        timing_profile: str = "default",
        debug_mode: bool = False,
        enable_metrics: bool = True,
    ):
        """
        Initialize conversation engine with all subsystems.

        Args:
            mai_interface: MaiInterface for model interaction
            memory_manager: MemoryManager for context management
            timing_profile: Timing profile ("default", "fast", "slow")
            debug_mode: Enable debug logging and verbose output
            enable_metrics: Enable performance metrics collection
        """
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.timing_profile = timing_profile
        self.debug_mode = debug_mode
        self.enable_metrics = enable_metrics

        # Initialize components
        self.mai_interface = mai_interface or MaiInterface()
        self.memory_manager = memory_manager or MemoryManager()

        # Conversation state management
        self.conversation_state = ConversationState()

        # Timing calculator for natural delays
        self.timing_calculator = TimingCalculator(profile=timing_profile)

        # Reasoning engine for step-by-step explanations
        self.reasoning_engine = ReasoningEngine()

        # Request decomposer for complex request analysis
        self.request_decomposer = RequestDecomposer()

        # Interruption handler for graceful recovery
        self.interrupt_handler = InterruptHandler()

        # Link conversation state with interrupt handler
        self.interrupt_handler.set_conversation_state(self.conversation_state)

        # Processing state for thread safety
        self.processing_threads: Dict[str, Thread] = {}
        self.interruption_events: Dict[str, Event] = {}
        self.current_processing: Dict[str, bool] = {}

        # Performance tracking
        self.total_conversations = 0
        self.total_interruptions = 0
        self.start_time = time.time()

        self.logger.info(
            f"ConversationEngine initialized with timing_profile='{timing_profile}', debug={debug_mode}"
        )

    def process_turn(
        self, user_message: str, conversation_id: Optional[str] = None
    ) -> ConversationResponse:
        """
        Process a single conversation turn with complete subsystem integration.

        Args:
            user_message: User's input message
            conversation_id: Optional conversation ID for continuation

        Returns:
            ConversationResponse with generated response and metadata
        """
        start_time = time.time()

        # Start or get conversation
        if conversation_id is None:
            conversation_id = self.conversation_state.start_conversation()
        else:
            conversation_id = self.conversation_state.start_conversation(conversation_id)

        # Handle interruption if already processing
        if self.conversation_state.is_processing(conversation_id):
            return self._handle_interruption(conversation_id, user_message, start_time)

        # Set processing lock
        self.conversation_state.set_processing(conversation_id, True)
        self.current_processing[conversation_id] = True

        try:
            self.logger.info(f"Processing conversation turn for {conversation_id}")

            # Check for reasoning request
            is_reasoning_request = self.reasoning_engine.is_reasoning_request(user_message)

            # Analyze request complexity and decomposition needs
            request_analysis = self.request_decomposer.analyze_request(user_message)

            # Handle clarification needs if request is ambiguous
            if request_analysis["needs_clarification"] and not is_reasoning_request:
                clarification_response = self._generate_clarification_response(request_analysis)
                return ConversationResponse(
                    response=clarification_response,
                    model_used="clarification",
                    tokens_used=0,
                    response_time=time.time() - start_time,
                    memory_context_used=0,
                    timing_category="clarification",
                    conversation_id=conversation_id,
                    interruption_handled=False,
                    memory_integrated=False,
                )

            # Retrieve memory context with 1000 token budget
            memory_context = self._retrieve_memory_context(user_message)

            # Build conversation history from state (last 10 turns)
            conversation_history = self.conversation_state.get_history(conversation_id)

            # Build memory-augmented prompt
            augmented_prompt = self._build_augmented_prompt(
                user_message, memory_context, conversation_history
            )

            # Calculate natural response delay based on cognitive load
            context_complexity = len(str(memory_context)) if memory_context else 0
            response_delay = self.timing_calculator.calculate_response_delay(
                user_message, context_complexity
            )

            # Apply natural delay for human-like interaction
            if not self.debug_mode:
                self.logger.info(f"Applying {response_delay:.2f}s delay for natural timing")
                time.sleep(response_delay)

            # Generate response with optional reasoning
            if is_reasoning_request:
                # Use reasoning engine for reasoning requests
                current_model = getattr(self.mai_interface, "current_model", "unknown")
                if current_model is None:
                    current_model = "unknown"
                reasoning_response = self.reasoning_engine.generate_response_with_reasoning(
                    user_message,
                    self.mai_interface.ollama_client,
                    current_model,
                    conversation_history,
                )
                interface_response = {
                    "response": reasoning_response["response"],
                    "model_used": reasoning_response["model_used"],
                    "tokens": reasoning_response.get("tokens_used", 0),
                    "response_time": response_delay,
                }
            else:
                # Standard response generation
                interface_response = self.mai_interface.send_message(
                    user_message, conversation_history
                )

            # Extract response details
            ai_response = interface_response.get(
                "response", "I apologize, but I couldn't generate a response."
            )
            model_used = interface_response.get("model_used", "unknown")
            tokens_used = interface_response.get("tokens", 0)

            # Store conversation turn in memory
            self._store_conversation_turn(
                conversation_id, user_message, ai_response, interface_response
            )

            # Create conversation turn with all metadata
            turn = ConversationTurn(
                conversation_id=conversation_id,
                user_message=user_message,
                ai_response=ai_response,
                timestamp=start_time,
                model_used=model_used,
                tokens_used=tokens_used,
                response_time=response_delay,
                memory_context_applied=bool(memory_context),
            )

            # Add turn to conversation state
            self.conversation_state.add_turn(turn)

            # Calculate response time and timing category
            total_response_time = time.time() - start_time
            complexity_score = self.timing_calculator.get_complexity_score(
                user_message, context_complexity
            )
            if complexity_score < 0.3:
                timing_category = "simple"
            elif complexity_score < 0.7:
                timing_category = "medium"
            else:
                timing_category = "complex"

            # Create comprehensive response object
            response = ConversationResponse(
                response=ai_response,
                model_used=model_used,
                tokens_used=tokens_used,
                response_time=total_response_time,
                memory_context_used=len(memory_context) if memory_context else 0,
                timing_category=timing_category,
                conversation_id=conversation_id,
                memory_integrated=bool(memory_context),
                interruption_handled=False,
            )

            self.total_conversations += 1
            self.logger.info(f"Conversation turn completed for {conversation_id}")

            return response

        except Exception as e:
            return ConversationResponse(
                response=f"I understand you want to move on. Let me help you with that.",
                model_used="error",
                tokens_used=0,
                response_time=time.time() - start_time,
                memory_context_used=0,
                timing_category="interruption",
                conversation_id=conversation_id,
                interruption_handled=True,
                memory_integrated=False,
            )

    def _generate_clarification_response(self, request_analysis: Dict[str, Any]) -> str:
        """
        Generate clarifying response for ambiguous requests.

        Args:
            request_analysis: Analysis from RequestDecomposer

        Returns:
            Clarifying response string
        """
        questions = request_analysis.get("clarification_questions", [])
        if not questions:
            return "Could you please provide more details about your request?"

        response_parts = ["I need some clarification to help you better:"]
        for i, question in enumerate(questions, 1):
            response_parts.append(f"{i}. {question}")

        response_parts.append("\nPlease provide the missing information and I'll be happy to help!")
        return "\n".join(response_parts)

    def _retrieve_memory_context(self, user_message: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve relevant memory context for user message.

        Uses 1000 token budget as specified in requirements.
        """
        try:
            if not self.memory_manager:
                return None

            # Get context with 1000 token budget and 3 max results
            context = self.memory_manager.get_context(
                query=user_message, max_tokens=1000, max_results=3
            )

            self.logger.debug(
                f"Retrieved {len(context.get('relevant_conversations', []))} relevant conversations"
            )
            return context

        except Exception as e:
            self.logger.warning(f"Failed to retrieve memory context: {e}")
            return None

    def _build_augmented_prompt(
        self,
        user_message: str,
        memory_context: Optional[Dict[str, Any]],
        conversation_history: List[Dict[str, str]],
    ) -> str:
        """
        Build memory-augmented prompt for model interaction.

        Integrates context and history as specified in requirements.
        """
        prompt_parts = []

        # Add memory context if available
        if memory_context and memory_context.get("relevant_conversations"):
            context_text = "Context from previous conversations:\n"
            for conv in memory_context["relevant_conversations"][:2]:  # Limit to 2 most relevant
                context_text += f"- {conv['title']}: {conv['excerpt']}\n"
            prompt_parts.append(context_text)

        # Add conversation history
        if conversation_history:
            history_text = "\nRecent conversation:\n"
            for msg in conversation_history[-10:]:  # Last 10 turns
                role = msg["role"]
                content = msg["content"][:200]  # Truncate long messages
                history_text += f"{role}: {content}\n"
            prompt_parts.append(history_text)

        # Add current user message
        prompt_parts.append(f"User: {user_message}")

        return "\n\n".join(prompt_parts)

    def _store_conversation_turn(
        self,
        conversation_id: str,
        user_message: str,
        ai_response: str,
        interface_response: Dict[str, Any],
    ) -> None:
        """
        Store conversation turn in memory using MemoryManager.

        Creates structured conversation data for persistence.
        """
        try:
            if not self.memory_manager:
                return

            # Build conversation messages for storage
            conversation_messages = []

            # Add context and history if available
            if interface_response.get("memory_context_used", 0) > 0:
                memory_context_msg = {
                    "role": "system",
                    "content": "Using memory context from previous conversations",
                }
                conversation_messages.append(memory_context_msg)

            # Add current turn
            conversation_messages.extend(
                [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": ai_response},
                ]
            )

            # Store in memory with metadata
            turn_metadata = {
                "conversation_id": conversation_id,
                "model_used": interface_response.get("model_used", "unknown"),
                "response_time": interface_response.get("response_time", 0),
                "tokens": interface_response.get("tokens", 0),
                "memory_context_applied": interface_response.get("memory_context_used", 0) > 0,
                "timestamp": time.time(),
                "engine_version": "conversation-engine-v1",
            }

            conv_id = self.memory_manager.store_conversation(
                messages=conversation_messages, metadata=turn_metadata
            )

            self.logger.debug(f"Stored conversation turn in memory: {conv_id}")

        except Exception as e:
            self.logger.warning(f"Failed to store conversation turn: {e}")

    def _handle_interruption(
        self, conversation_id: str, new_message: str, start_time: float
    ) -> ConversationResponse:
        """
        Handle user interruption during processing.

        Clears pending response and restarts with new context using InterruptHandler.
        """
        self.logger.info(f"Handling interruption for conversation {conversation_id}")
        self.total_interruptions += 1

        # Create interruption context
        interrupt_context = self.interrupt_handler.interrupt_and_restart(
            new_message=new_message,
            conversation_id=conversation_id,
            turn_type=TurnType.USER_INPUT,
            reason="user_input",
        )

        # Restart processing with new message (immediate response for interruption)
        try:
            interface_response = self.mai_interface.send_message(
                new_message, self.conversation_state.get_history(conversation_id)
            )

            return ConversationResponse(
                response=interface_response.get(
                    "response", "I understand you want to move on. How can I help you?"
                ),
                model_used=interface_response.get("model_used", "unknown"),
                tokens_used=interface_response.get("tokens", 0),
                response_time=time.time() - start_time,
                memory_context_used=0,
                timing_category="interruption",
                conversation_id=conversation_id,
                interruption_handled=True,
                memory_integrated=False,
            )

        except Exception as e:
            return ConversationResponse(
                response=f"I understand you want to move on. Let me help you with that.",
                model_used="error",
                tokens_used=0,
                response_time=time.time() - start_time,
                memory_context_used=0,
                timing_category="interruption",
                conversation_id=conversation_id,
                interruption_handled=True,
                memory_integrated=False,
            )

    def get_conversation_history(
        self, conversation_id: str, limit: int = 10
    ) -> List[ConversationTurn]:
        """Get conversation history for a specific conversation."""
        return self.conversation_state.get_conversation_turns(conversation_id)[-limit:]

    def get_engine_stats(self) -> Dict[str, Any]:
        """Get engine performance statistics."""
        uptime = time.time() - self.start_time

        return {
            "uptime_seconds": uptime,
            "total_conversations": self.total_conversations,
            "total_interruptions": self.total_interruptions,
            "active_conversations": len(self.conversation_state.conversations),
            "average_response_time": 0.0,  # Would be calculated from actual responses
            "memory_integration_rate": 0.0,  # Would be calculated from actual responses
        }

    def calculate_response_delay(
        self, user_message: str, context_complexity: Optional[int] = None
    ) -> float:
        """
        Calculate natural response delay using TimingCalculator.

        Args:
            user_message: User message to analyze
            context_complexity: Optional context complexity

        Returns:
            Response delay in seconds
        """
        return self.timing_calculator.calculate_response_delay(user_message, context_complexity)

    def is_reasoning_request(self, user_message: str) -> bool:
        """
        Check if user is requesting reasoning explanation.

        Args:
            user_message: User message to analyze

        Returns:
            True if this appears to be a reasoning request
        """
        return self.reasoning_engine.is_reasoning_request(user_message)

    def generate_response_with_reasoning(
        self, user_message: str, conversation_history: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Generate response with step-by-step reasoning explanation.

        Args:
            user_message: Original user message
            conversation_history: Conversation context

        Returns:
            Dictionary with reasoning-enhanced response
        """
        current_model = getattr(self.mai_interface, "current_model", "unknown")
        if current_model is None:
            current_model = "unknown"

        return self.reasoning_engine.generate_response_with_reasoning(
            user_message, self.mai_interface.ollama_client, current_model, conversation_history
        )

    def analyze_request_complexity(self, user_message: str) -> Dict[str, Any]:
        """
        Analyze request complexity and decomposition needs.

        Args:
            user_message: User message to analyze

        Returns:
            Request analysis dictionary
        """
        return self.request_decomposer.analyze_request(user_message)

    def check_interruption(self, conversation_id: str) -> bool:
        """
        Check if interruption has occurred for a conversation.

        Args:
            conversation_id: ID of conversation to check

        Returns:
            True if interruption detected
        """
        return self.interrupt_handler.check_interruption(conversation_id)

    def interrupt_and_restart(
        self, new_message: str, conversation_id: str, reason: str = "user_input"
    ) -> Dict[str, Any]:
        """
        Handle interruption and restart conversation.

        Args:
            new_message: New message that triggered interruption
            conversation_id: ID of conversation
            reason: Reason for interruption

        Returns:
            Interruption context dictionary
        """
        interrupt_context = self.interrupt_handler.interrupt_and_restart(
            new_message=new_message,
            conversation_id=conversation_id,
            turn_type=TurnType.USER_INPUT,
            reason=reason,
        )
        return interrupt_context.to_dict()

    def needs_clarification(self, request_analysis: Dict[str, Any]) -> bool:
        """
        Check if request needs clarification.

        Args:
            request_analysis: Request analysis result

        Returns:
            True if clarification is needed
        """
        return request_analysis.get("needs_clarification", False)

    def suggest_breakdown(self, user_message: str, complexity_score: float) -> Dict[str, Any]:
        """
        Suggest logical breakdown for complex requests.

        Args:
            user_message: Original user message
            complexity_score: Complexity score from analysis

        Returns:
            Breakdown suggestions dictionary
        """
        return self.request_decomposer.suggest_breakdown(
            user_message,
            complexity_score,
            self.mai_interface.ollama_client,
            getattr(self.mai_interface, "current_model", "default"),
        )

    def adapt_response_with_personality(
        self, response: str, user_message: str, context_type: Optional[str] = None
    ) -> str:
        """
        Adapt response based on personality guidelines.

        Args:
            response: Generated response to adapt
            user_message: Original user message for context
            context_type: Type of conversation context

        Returns:
            Personality-adapted response
        """
        # For now, return original response
        # Personality integration will be implemented in Phase 9
        return response

    def cleanup(self, max_age_hours: int = 24) -> None:
        """Clean up old conversations and resources."""
        self.conversation_state.cleanup_old_conversations(max_age_hours)
        self.logger.info(f"Cleaned up conversations older than {max_age_hours} hours")

    def shutdown(self) -> None:
        """Shutdown conversation engine gracefully."""
        self.logger.info("Shutting down ConversationEngine...")

        # Cancel any processing threads
        for conv_id, thread in self.processing_threads.items():
            if thread.is_alive():
                if conv_id in self.interruption_events:
                    self.interruption_events[conv_id].set()
                thread.join(timeout=1.0)

        # Cleanup resources
        self.cleanup()

        self.logger.info("ConversationEngine shutdown complete")

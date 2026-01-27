"""
Reasoning Transparency Engine for Mai

Provides step-by-step reasoning explanations when explicitly requested
by users, with caching for performance optimization.
"""

import logging
import hashlib
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ReasoningEngine:
    """
    Provides reasoning transparency and step-by-step explanations.

    This engine detects when users explicitly ask for reasoning explanations
    and generates detailed step-by-step breakdowns of Mai's thought process.
    """

    def __init__(self):
        """Initialize reasoning engine with caching."""
        self.logger = logging.getLogger(__name__)

        # Cache for reasoning explanations to avoid recomputation
        self._reasoning_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_duration = timedelta(hours=24)

        # Keywords that indicate reasoning requests
        self._reasoning_keywords = [
            "how did you",
            "explain your reasoning",
            "step by step",
            "why",
            "process",
            "how do you know",
            "what makes you think",
            "show your work",
            "walk through",
            "break down",
            "explain your logic",
            "how did you arrive",
            "what's your reasoning",
            "explain yourself",
        ]

        self.logger.info("ReasoningEngine initialized")

    def is_reasoning_request(self, message: str) -> bool:
        """
        Detect when user explicitly asks for reasoning explanation.

        Args:
            message: User message to analyze

        Returns:
            True if this appears to be a reasoning request
        """
        message_lower = message.lower().strip()

        # Check for reasoning keywords
        for keyword in self._reasoning_keywords:
            if keyword in message_lower:
                self.logger.debug(f"Reasoning request detected via keyword: {keyword}")
                return True

        # Check for question patterns asking about process
        reasoning_patterns = [
            r"how did you",
            r"why.*you.*\?",
            r"what.*your.*process",
            r"can you.*explain.*your",
            r"show.*your.*work",
            r"explain.*how.*you",
            r"what.*your.*reasoning",
            r"walk.*through.*your",
        ]

        import re

        for pattern in reasoning_patterns:
            if re.search(pattern, message_lower):
                self.logger.debug(f"Reasoning request detected via pattern: {pattern}")
                return True

        return False

    def _get_cache_key(self, message: str) -> str:
        """Generate cache key based on message content hash."""
        return hashlib.md5(message.encode()).hexdigest()

    def _is_cache_valid(self, cache_entry: Optional[Dict[str, Any]]) -> bool:
        """Check if cache entry is still valid."""
        if not cache_entry:
            return False

        cached_time = cache_entry.get("timestamp")
        if not cached_time:
            return False

        return datetime.now() - cached_time < self._cache_duration

    def generate_response_with_reasoning(
        self,
        user_message: str,
        ollama_client,
        current_model: str,
        context: Optional[List[Dict[str, Any]]] = None,
        show_reasoning: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate response with optional step-by-step reasoning explanation.

        Args:
            user_message: Original user message
            ollama_client: OllamaClient instance for generating responses
            current_model: Current model name
            context: Conversation context
            show_reasoning: Whether to include reasoning explanation

        Returns:
            Dictionary with response, reasoning (if requested), and metadata
        """
        # Check cache first
        cache_key = self._get_cache_key(user_message)
        cached_entry = self._reasoning_cache.get(cache_key)

        if (
            cached_entry
            and self._is_cache_valid(cached_entry)
            and cached_entry.get("message") == user_message
        ):
            self.logger.debug("Using cached reasoning response")
            return cached_entry["response"]

        # Detect if this is a reasoning request
        is_reasoning = show_reasoning or self.is_reasoning_request(user_message)

        try:
            # Generate standard response
            standard_response = ollama_client.generate_response(
                user_message, current_model, context or []
            )

            response_data = {
                "response": standard_response,
                "model_used": current_model,
                "show_reasoning": is_reasoning,
                "reasoning": None,
                "format": "standard",
            }

            # Generate reasoning explanation if requested
            if is_reasoning:
                reasoning = self._generate_reasoning_explanation(
                    user_message, standard_response, ollama_client, current_model
                )
                response_data["reasoning"] = reasoning
                response_data["format"] = "with_reasoning"

                # Format response with reasoning
                formatted_response = self.format_reasoning_response(standard_response, reasoning)
                response_data["response"] = formatted_response

            # Cache the response
            self._reasoning_cache[cache_key] = {
                "message": user_message,
                "response": response_data,
                "timestamp": datetime.now(),
            }

            self.logger.info(f"Generated response with reasoning={is_reasoning}")
            return response_data

        except Exception as e:
            self.logger.error(f"Failed to generate response with reasoning: {e}")
            raise

    def _generate_reasoning_explanation(
        self, user_message: str, standard_response: str, ollama_client, current_model: str
    ) -> str:
        """
        Generate step-by-step reasoning explanation.

        Args:
            user_message: Original user question
            standard_response: The response that was generated
            ollama_client: OllamaClient for generating reasoning
            current_model: Current model name

        Returns:
            Formatted reasoning explanation as numbered steps
        """
        reasoning_prompt = f"""
Explain your reasoning step by step for answering: "{user_message}"

Your final answer was: "{standard_response}"

Please explain your reasoning process:
1. Start by understanding what the user is asking
2. Break down the key components of the question
3. Explain your thought process step by step
4. Show how you arrived at your conclusion
5. End with "Final answer:" followed by your actual response

Format as clear numbered steps. Be detailed but concise.
"""

        try:
            reasoning = ollama_client.generate_response(reasoning_prompt, current_model, [])
            return self._clean_reasoning_output(reasoning)

        except Exception as e:
            self.logger.error(f"Failed to generate reasoning explanation: {e}")
            return f"I apologize, but I encountered an error generating my reasoning explanation. My response was: {standard_response}"

    def _clean_reasoning_output(self, reasoning: str) -> str:
        """Clean and format reasoning output."""
        # Remove any redundant prefixes
        reasoning = reasoning.strip()

        # Remove common AI response prefixes
        prefixes_to_remove = [
            "Here's my reasoning:",
            "My reasoning is:",
            "Let me explain my reasoning:",
            "I'll explain my reasoning step by step:",
        ]

        for prefix in prefixes_to_remove:
            if reasoning.startswith(prefix):
                reasoning = reasoning[len(prefix) :].strip()
                break

        return reasoning

    def format_reasoning_response(self, response: str, reasoning: str) -> str:
        """
        Format reasoning with clear separation from main answer.

        Args:
            response: The actual response
            reasoning: The reasoning explanation

        Returns:
            Formatted response with reasoning section
        """
        # Clean up any existing formatting
        reasoning = self._clean_reasoning_output(reasoning)

        # Format with clear separation
        formatted = f"""## 🧠 My Reasoning Process

{reasoning}

---
## 💬 My Response

{response}"""

        return formatted

    def clear_cache(self) -> None:
        """Clear reasoning cache."""
        self._reasoning_cache.clear()
        self.logger.info("Reasoning cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get reasoning cache statistics."""
        total_entries = len(self._reasoning_cache)
        valid_entries = sum(
            1 for entry in self._reasoning_cache.values() if self._is_cache_valid(entry)
        )

        return {
            "total_entries": total_entries,
            "valid_entries": valid_entries,
            "cache_duration_hours": self._cache_duration.total_seconds() / 3600,
            "last_cleanup": datetime.now().isoformat(),
        }

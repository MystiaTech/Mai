"""
Natural Timing Calculation for Mai

Provides human-like response delays based on cognitive load analysis
with natural variation to avoid robotic consistency.
"""

import time
import random
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class TimingCalculator:
    """
    Calculates natural response delays based on cognitive load analysis.

    Generates human-like timing variation considering message complexity,
    question count, technical content, and context depth.
    """

    def __init__(self, profile: str = "default"):
        """
        Initialize timing calculator with specified profile.

        Args:
            profile: Timing profile - "default", "fast", or "slow"
        """
        self.profile = profile
        self.logger = logging.getLogger(__name__)

        # Profile-specific multipliers
        self.profiles = {
            "default": {"base": 1.0, "variation": 0.3},
            "fast": {"base": 0.6, "variation": 0.2},
            "slow": {"base": 1.4, "variation": 0.4},
        }

        if profile not in self.profiles:
            self.logger.warning(f"Unknown profile '{profile}', using 'default'")
            self.profile = "default"

        self.profile_config = self.profiles[self.profile]
        self.logger.info(f"TimingCalculator initialized with '{self.profile}' profile")

    def calculate_response_delay(
        self, message: str, context_complexity: Optional[int] = None
    ) -> float:
        """
        Calculate natural response delay based on cognitive load.

        Args:
            message: User message to analyze
            context_complexity: Optional context complexity score

        Returns:
            Response delay in seconds (1.0-8.0 range)
        """
        # Analyze message complexity
        complexity_score = self.get_complexity_score(message, context_complexity)

        # Determine base delay based on complexity category
        if complexity_score < 0.3:
            # Simple (low complexity)
            base_delay = random.uniform(1.5, 2.5)
            category = "simple"
        elif complexity_score < 0.7:
            # Medium (moderate complexity)
            base_delay = random.uniform(2.0, 4.0)
            category = "medium"
        else:
            # Complex (high complexity)
            base_delay = random.uniform(3.0, 8.0)
            category = "complex"

        # Apply profile multiplier
        adjusted_delay = base_delay * self.profile_config["base"]

        # Add natural variation/jitter
        variation_amount = adjusted_delay * self.profile_config["variation"]
        jitter = random.uniform(-0.2, 0.2)  # +/-0.2 seconds
        final_delay = max(0.5, adjusted_delay + variation_amount + jitter)  # Minimum 0.5s

        self.logger.debug(
            f"Delay calculation: {category} complexity ({complexity_score:.2f}) -> {final_delay:.2f}s"
        )

        # Ensure within reasonable bounds
        return min(max(final_delay, 0.5), 10.0)  # 0.5s to 10s range

    def get_complexity_score(self, message: str, context_complexity: Optional[int] = None) -> float:
        """
        Analyze message content for complexity indicators.

        Args:
            message: Message to analyze
            context_complexity: Optional context complexity from conversation history

        Returns:
            Complexity score from 0.0 (simple) to 1.0 (complex)
        """
        score = 0.0

        # 1. Message length factor (0-0.3)
        word_count = len(message.split())
        if word_count > 50:
            score += 0.3
        elif word_count > 25:
            score += 0.2
        elif word_count > 10:
            score += 0.1

        # 2. Question count factor (0-0.3)
        question_count = message.count("?")
        if question_count >= 3:
            score += 0.3
        elif question_count >= 2:
            score += 0.2
        elif question_count >= 1:
            score += 0.1

        # 3. Technical content indicators (0-0.3)
        technical_keywords = [
            "function",
            "class",
            "algorithm",
            "debug",
            "implement",
            "fix",
            "error",
            "optimization",
            "performance",
            "database",
            "api",
            "endpoint",
            "method",
            "parameter",
            "variable",
            "constant",
            "import",
            "export",
            "async",
            "await",
            "promise",
            "callback",
            "recursive",
            "iterative",
            "hash",
            "encryption",
            "authentication",
            "authorization",
            "token",
            "session",
        ]

        technical_count = sum(
            1 for keyword in technical_keywords if keyword.lower() in message.lower()
        )
        if technical_count >= 5:
            score += 0.3
        elif technical_count >= 3:
            score += 0.2
        elif technical_count >= 1:
            score += 0.1

        # 4. Code pattern indicators (0-0.2)
        code_indicators = 0
        if "```" in message:
            code_indicators += 1
        if "`" in message and message.count("`") >= 2:
            code_indicators += 1
        if any(
            word in message.lower() for word in ["def", "function", "class", "var", "let", "const"]
        ):
            code_indicators += 1
        if any(char in message for char in ["{}()\[\];"]):
            code_indicators += 1

        if code_indicators >= 1:
            score += 0.1
            if code_indicators >= 2:
                score += 0.1

        # 5. Context complexity integration (0-0.2)
        if context_complexity is not None:
            if context_complexity > 1000:  # High token context
                score += 0.2
            elif context_complexity > 500:  # Medium token context
                score += 0.1

        # Normalize to 0-1 range
        normalized_score = min(score, 1.0)

        self.logger.debug(
            f"Complexity analysis: score={normalized_score:.2f}, words={word_count}, questions={question_count}, technical={technical_count}"
        )

        return normalized_score

    def set_profile(self, profile: str) -> None:
        """
        Change timing profile.

        Args:
            profile: New profile name ("default", "fast", "slow")
        """
        if profile in self.profiles:
            self.profile = profile
            self.profile_config = self.profiles[profile]
            self.logger.info(f"Timing profile changed to '{profile}'")
        else:
            self.logger.warning(
                f"Unknown profile '{profile}', keeping current profile '{self.profile}'"
            )

    def get_timing_stats(self, messages: list) -> Dict[str, Any]:
        """
        Calculate timing statistics for a list of messages.

        Args:
            messages: List of message dictionaries with timing info

        Returns:
            Dictionary with timing statistics
        """
        if not messages:
            return {
                "message_count": 0,
                "average_delay": 0.0,
                "min_delay": 0.0,
                "max_delay": 0.0,
                "total_delay": 0.0,
            }

        delays = []
        total_delay = 0.0

        for msg in messages:
            if "response_time" in msg:
                delays.append(msg["response_time"])
                total_delay += msg["response_time"]

        if delays:
            return {
                "message_count": len(messages),
                "average_delay": total_delay / len(delays),
                "min_delay": min(delays),
                "max_delay": max(delays),
                "total_delay": total_delay,
                "profile": self.profile,
            }
        else:
            return {
                "message_count": len(messages),
                "average_delay": 0.0,
                "min_delay": 0.0,
                "max_delay": 0.0,
                "total_delay": 0.0,
                "profile": self.profile,
            }

    def get_profile_info(self) -> Dict[str, Any]:
        """
        Get information about current timing profile.

        Returns:
            Dictionary with profile configuration
        """
        return {
            "current_profile": self.profile,
            "base_multiplier": self.profile_config["base"],
            "variation_range": self.profile_config["variation"],
            "available_profiles": list(self.profiles.keys()),
            "description": {
                "default": "Natural human-like timing with moderate variation",
                "fast": "Reduced delays for quick interactions and testing",
                "slow": "Extended delays for thoughtful, deliberate responses",
            }.get(self.profile, "Unknown profile"),
        }

"""
Request Decomposition and Clarification Engine for Mai

Analyzes request complexity and generates appropriate clarifying questions
when user requests are ambiguous or overly complex.
"""

import logging
import re
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)


class RequestDecomposer:
    """
    Analyzes request complexity and generates clarifying questions.

    This engine identifies ambiguous requests, assesses complexity,
    and generates specific clarifying questions to improve understanding.
    """

    def __init__(self):
        """Initialize request decomposer with analysis patterns."""
        self.logger = logging.getLogger(__name__)

        # Ambiguity patterns to detect
        self._ambiguity_patterns = {
            "pronouns_without_antecedents": [
                r"\b(it|that|this|they|them|these|those)\b",
                r"\b(he|she|it)\s+(?:is|was|were|will|would|could|should)",
            ],
            "vague_quantifiers": [
                r"\b(some|few|many|several|multiple|various|better|faster|more|less)\b",
                r"\b(a bit|a little|quite|very|really|somewhat)\b",
            ],
            "missing_context": [
                r"\b(the|that|this|there)\s+(?:here|there)",
                r"\b(?:from|about|regarding|concerning)\s+(?:it|that|this)",
            ],
            "undefined_references": [
                r"\b(?:fix|improve|update|change|modify)\s+(?:it|that|this)",
                r"\b(?:do|make|create|build)\s+(?:it|that|this)",
            ],
        }

        # Complexity indicators
        self._complexity_indicators = {
            "technical_keywords": [
                "function",
                "algorithm",
                "database",
                "api",
                "class",
                "method",
                "variable",
                "loop",
                "conditional",
                "recursion",
                "optimization",
                "debug",
                "implement",
                "integrate",
                "configure",
                "deploy",
            ],
            "multiple_tasks": [
                r"\band\b",
                r"\bthen\b",
                r"\bafter\b",
                r"\balso\b",
                r"\bnext\b",
                r"\bfinally\b",
                r"\badditionally\b",
            ],
            "question_density": r"[?！]",
            "length_threshold": 150,  # characters
        }

        self.logger.info("RequestDecomposer initialized")

    def analyze_request(self, message: str) -> Dict[str, Any]:
        """
        Analyze request for complexity and ambiguity.

        Args:
            message: User message to analyze

        Returns:
            Dictionary with analysis results including:
            - needs_clarification: boolean
            - complexity_score: float (0-1)
            - estimated_steps: int
            - clarification_questions: list
            - ambiguity_indicators: list
        """
        message_lower = message.lower().strip()

        # Detect ambiguities
        ambiguity_indicators = self._detect_ambiguities(message_lower)
        needs_clarification = len(ambiguity_indicators) > 0

        # Calculate complexity score
        complexity_score = self._calculate_complexity(message)

        # Estimate steps needed
        estimated_steps = self._estimate_steps(message, complexity_score)

        # Generate clarification questions
        clarification_questions = []
        if needs_clarification:
            clarification_questions = self._generate_clarifications(message, ambiguity_indicators)

        return {
            "needs_clarification": needs_clarification,
            "complexity_score": complexity_score,
            "estimated_steps": estimated_steps,
            "clarification_questions": clarification_questions,
            "ambiguity_indicators": ambiguity_indicators,
            "message_length": len(message),
            "word_count": len(message.split()),
        }

    def _detect_ambiguities(self, message: str) -> List[Dict[str, Any]]:
        """
        Detect specific ambiguity indicators in the message.

        Args:
            message: Lowercase message to analyze

        Returns:
            List of ambiguity indicators with details
        """
        ambiguities = []

        for category, patterns in self._ambiguity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, message, re.IGNORECASE)
                for match in matches:
                    ambiguities.append(
                        {
                            "type": category,
                            "pattern": pattern,
                            "match": match.group(),
                            "position": match.start(),
                            "context": self._get_context(message, match.start(), match.end()),
                        }
                    )

        return ambiguities

    def _get_context(self, message: str, start: int, end: int, window: int = 20) -> str:
        """Get context around a match."""
        context_start = max(0, start - window)
        context_end = min(len(message), end + window)
        return message[context_start:context_end]

    def _calculate_complexity(self, message: str) -> float:
        """
        Calculate complexity score based on multiple factors.

        Args:
            message: Message to analyze

        Returns:
            Complexity score between 0.0 (simple) and 1.0 (complex)
        """
        complexity = 0.0

        # Technical content (0.3 weight)
        technical_count = sum(
            1
            for keyword in self._complexity_indicators["technical_keywords"]
            if keyword.lower() in message.lower()
        )
        technical_score = min(technical_count * 0.1, 0.3)
        complexity += technical_score

        # Multiple tasks (0.25 weight)
        task_matches = 0
        for pattern in self._complexity_indicators["multiple_tasks"]:
            matches = len(re.findall(pattern, message, re.IGNORECASE))
            task_matches += matches
        task_score = min(task_matches * 0.08, 0.25)
        complexity += task_score

        # Question density (0.2 weight)
        question_count = len(re.findall(self._complexity_indicators["question_density"], message))
        question_score = min(question_count * 0.05, 0.2)
        complexity += question_score

        # Message length (0.15 weight)
        length_score = min(len(message) / 500, 0.15)
        complexity += length_score

        # Sentence complexity (0.1 weight)
        sentences = message.split(".")
        avg_sentence_length = sum(len(s.strip()) for s in sentences if s.strip()) / max(
            len(sentences), 1
        )
        sentence_score = min(avg_sentence_length / 100, 0.1)
        complexity += sentence_score

        return min(complexity, 1.0)

    def _estimate_steps(self, message: str, complexity_score: float) -> int:
        """
        Estimate number of steps needed to fulfill request.

        Args:
            message: Original message
            complexity_score: Calculated complexity score

        Returns:
            Estimated number of steps
        """
        base_steps = 1

        # Add steps for multiple tasks
        task_count = 0
        for pattern in self._complexity_indicators["multiple_tasks"]:
            matches = len(re.findall(pattern, message, re.IGNORECASE))
            task_count += matches
        base_steps += max(0, task_count - 1)  # First task is step 1

        # Add steps for complexity
        if complexity_score > 0.7:
            base_steps += 3  # Complex requests need planning
        elif complexity_score > 0.5:
            base_steps += 2  # Medium complexity needs some breakdown
        elif complexity_score > 0.3:
            base_steps += 1  # Slightly complex might need clarification

        return max(1, base_steps)

    def _generate_clarifications(
        self, message: str, ambiguity_indicators: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Generate specific clarifying questions for detected ambiguities.

        Args:
            message: Original message
            ambiguity_indicators: List of detected ambiguities

        Returns:
            List of clarifying questions
        """
        questions = []
        seen_types = set()

        for indicator in ambiguity_indicators:
            ambiguity_type = indicator["type"]
            match = indicator["match"]

            # Avoid duplicate questions for same ambiguity type
            if ambiguity_type in seen_types:
                continue
            seen_types.add(ambiguity_type)

            if ambiguity_type == "pronouns_without_antecedents":
                if match.lower() in ["it", "that", "this"]:
                    questions.append(f"Could you clarify what '{match}' refers to specifically?")
                elif match.lower() in ["they", "them", "these", "those"]:
                    questions.append(f"Could you specify who or what '{match}' refers to?")

            elif ambiguity_type == "vague_quantifiers":
                if match.lower() in ["better", "faster", "more", "less"]:
                    questions.append(f"Could you quantify what '{match}' means in this context?")
                elif match.lower() in ["some", "few", "many", "several"]:
                    questions.append(
                        f"Could you provide a specific number or amount instead of '{match}'?"
                    )
                else:
                    questions.append(f"Could you be more specific about what '{match}' means?")

            elif ambiguity_type == "missing_context":
                questions.append(f"Could you provide more context about what '{match}' refers to?")

            elif ambiguity_type == "undefined_references":
                questions.append(f"Could you clarify what you'd like me to {match} specifically?")

        return questions

    def suggest_breakdown(
        self,
        message: str,
        complexity_score: float,
        ollama_client=None,
        current_model: str = "default",
    ) -> Dict[str, Any]:
        """
        Suggest logical breakdown for complex requests.

        Args:
            message: Original user message
            complexity_score: Calculated complexity
            ollama_client: Optional OllamaClient for semantic analysis
            current_model: Current model name

        Returns:
            Dictionary with breakdown suggestions
        """
        estimated_steps = self._estimate_steps(message, complexity_score)

        # Extract potential tasks from message
        tasks = self._extract_tasks(message)

        breakdown = {
            "estimated_steps": estimated_steps,
            "complexity_level": self._get_complexity_level(complexity_score),
            "suggested_approach": [],
            "potential_tasks": tasks,
            "effort_estimate": self._estimate_effort(complexity_score),
        }

        # Generate approach suggestions
        if complexity_score > 0.6:
            breakdown["suggested_approach"].append(
                "Start by clarifying requirements and breaking into smaller tasks"
            )
            breakdown["suggested_approach"].append(
                "Consider if this needs to be done in sequence or can be parallelized"
            )
        elif complexity_score > 0.3:
            breakdown["suggested_approach"].append(
                "Break down into logical sub-tasks before starting"
            )

        # Use semantic analysis if available and request is very complex
        if ollama_client and complexity_score > 0.7:
            try:
                semantic_breakdown = self._semantic_breakdown(message, ollama_client, current_model)
                breakdown["semantic_analysis"] = semantic_breakdown
            except Exception as e:
                self.logger.warning(f"Semantic breakdown failed: {e}")

        return breakdown

    def _extract_tasks(self, message: str) -> List[str]:
        """Extract potential tasks from message."""
        # Simple task extraction based on verbs and patterns
        task_patterns = [
            r"(?:please\s+)?(?:can\s+you\s+)?(\w+)\s+(.+?)(?:\s+(?:and|then|after)\s+|$)",
            r"(?:I\s+need|want)\s+(?:you\s+to\s+)?(.+?)(?:\s+(?:and|then|after)\s+|$)",
            r"(?:help\s+me\s+)?(\w+)\s+(.+?)(?:\s+(?:and|then|after)\s+|$)",
        ]

        tasks = []
        for pattern in task_patterns:
            matches = re.findall(pattern, message, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    # Take the verb/object combination
                    task = " ".join(filter(None, match))
                else:
                    task = str(match)
                if len(task.strip()) > 3:  # Filter out very short matches
                    tasks.append(task.strip())

        return list(set(tasks))  # Remove duplicates

    def _get_complexity_level(self, score: float) -> str:
        """Convert complexity score to human-readable level."""
        if score >= 0.7:
            return "High"
        elif score >= 0.4:
            return "Medium"
        else:
            return "Low"

    def _estimate_effort(self, complexity_score: float) -> str:
        """Estimate effort based on complexity."""
        if complexity_score >= 0.7:
            return "Significant - may require multiple iterations"
        elif complexity_score >= 0.4:
            return "Moderate - should be straightforward with some planning"
        else:
            return "Minimal - should be quick to implement"

    def _semantic_breakdown(self, message: str, ollama_client, current_model: str) -> str:
        """
        Use AI to perform semantic breakdown of complex request.

        Args:
            message: User message to analyze
            ollama_client: OllamaClient instance
            current_model: Current model name

        Returns:
            AI-generated breakdown suggestions
        """
        semantic_prompt = f"""
Analyze this complex request and suggest a logical breakdown: "{message}"

Provide a structured approach:
1. Identify the main objectives
2. Break down into logical steps
3. Note any dependencies or prerequisites
4. Suggest an order of execution

Keep it concise and actionable.
"""

        try:
            response = ollama_client.generate_response(semantic_prompt, current_model, [])
            return self._clean_semantic_output(response)
        except Exception as e:
            self.logger.error(f"Semantic breakdown failed: {e}")
            return "Unable to generate semantic breakdown"

    def _clean_semantic_output(self, output: str) -> str:
        """Clean semantic breakdown output."""
        # Remove common AI response prefixes
        prefixes_to_remove = [
            "Here's a breakdown:",
            "Let me break this down:",
            "I would approach this by:",
            "Here's how I would break this down:",
        ]

        for prefix in prefixes_to_remove:
            if output.startswith(prefix):
                output = output[len(prefix) :].strip()
                break

        return output

    def get_analysis_summary(self, analysis: Dict[str, Any]) -> str:
        """
        Get human-readable summary of request analysis.

        Args:
            analysis: Result from analyze_request()

        Returns:
            Formatted summary string
        """
        summary_parts = []

        if analysis["needs_clarification"]:
            summary_parts.append("🤔 **Needs Clarification**")
            summary_parts.append(f"- Questions: {len(analysis['clarification_questions'])}")
        else:
            summary_parts.append("✅ **Clear Request**")

        complexity_level = self._get_complexity_level(analysis["complexity_score"])
        summary_parts.append(
            f"📊 **Complexity**: {complexity_level} ({analysis['complexity_score']:.2f})"
        )
        summary_parts.append(f"📋 **Estimated Steps**: {analysis['estimated_steps']}")

        if analysis["ambiguity_indicators"]:
            summary_parts.append(
                f"⚠️ **Ambiguities Found**: {len(analysis['ambiguity_indicators'])}"
            )

        return "\n".join(summary_parts)

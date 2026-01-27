"""
Model selection and switching logic for Mai.

Intelligently selects and switches between models based on
available resources and conversation requirements.
"""

import time
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum


class ModelSize(Enum):
    """Model size categories"""

    TINY = "1b"
    SMALL = "3b"
    MEDIUM = "7b"
    LARGE = "13b"
    HUGE = "70b"


class SwitchReason(Enum):
    """Reasons for model switching"""

    RESOURCE_CONSTRAINT = "resource_constraint"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    TASK_COMPLEXITY = "task_complexity"
    USER_REQUEST = "user_request"
    ERROR_RECOVERY = "error_recovery"
    PROACTIVE_OPTIMIZATION = "proactive_optimization"


@dataclass
class ModelInfo:
    """Information about an available model"""

    name: str
    size: str  # '7b', '13b', etc.
    parameters: int  # parameter count
    context_window: int  # context window size
    quantization: str  # 'q4_0', 'q8_0', etc.
    modified_at: Optional[str] = None
    digest: Optional[str] = None

    def __post_init__(self):
        """Post-processing for model info"""
        # Extract parameter count from size string
        if isinstance(self.size, str):
            import re

            match = re.search(r"(\d+\.?\d*)", self.size.lower())
            if match:
                self.parameters = int(float(match.group(1)) * 1e9)

        # Determine size category
        if self.parameters <= 2e9:
            self.size_category = ModelSize.TINY
        elif self.parameters <= 4e9:
            self.size_category = ModelSize.SMALL
        elif self.parameters <= 10e9:
            self.size_category = ModelSize.MEDIUM
        elif self.parameters <= 20e9:
            self.size_category = ModelSize.LARGE
        else:
            self.size_category = ModelSize.HUGE
        self.size_category = ModelSize.MEDIUM  # Default override for now


@dataclass
class SwitchMetrics:
    """Metrics for model switching performance"""

    switch_time: float
    context_transfer_time: float
    context_compression_ratio: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class SwitchRecommendation:
    """Recommendation for model switching"""

    should_switch: bool
    target_model: Optional[str]
    reason: SwitchReason
    confidence: float
    expected_benefit: str
    estimated_cost: Dict[str, float]


class ModelSwitcher:
    """Intelligent model selection and switching"""

    def __init__(self, ollama_client, resource_detector):
        """Initialize model switcher with dependencies"""
        self.client = ollama_client
        self.resource_detector = resource_detector

        # Current state
        self.current_model: Optional[str] = None
        self.current_model_info: Optional[ModelInfo] = None
        self.conversation_context: List[Dict] = []

        # Switching history and performance
        self.switch_history: List[Dict] = []
        self.performance_metrics: Dict[str, Any] = {
            "switch_count": 0,
            "successful_switches": 0,
            "average_switch_time": 0.0,
            "last_switch_time": None,
        }

        # Model capability mappings
        self.model_requirements = {
            ModelSize.TINY: {"memory_gb": 2, "cpu_cores": 2, "context_preference": 0.3},
            ModelSize.SMALL: {"memory_gb": 4, "cpu_cores": 4, "context_preference": 0.5},
            ModelSize.MEDIUM: {"memory_gb": 8, "cpu_cores": 6, "context_preference": 0.7},
            ModelSize.LARGE: {"memory_gb": 16, "cpu_cores": 8, "context_preference": 0.8},
            ModelSize.HUGE: {"memory_gb": 80, "cpu_cores": 16, "context_preference": 1.0},
        }

        # Initialize available models
        self.available_models: Dict[str, ModelInfo] = {}
        # Models will be refreshed when needed
        # Note: _refresh_model_list is async and should be called from async context

    async def _refresh_model_list(self):
        """Refresh the list of available models"""
        try:
            # This would use the actual ollama client
            # For now, create mock models
            self.available_models = {
                "llama3.2:1b": ModelInfo(
                    name="llama3.2:1b",
                    size="1b",
                    parameters=1_000_000_000,
                    context_window=2048,
                    quantization="q4_0",
                ),
                "llama3.2:3b": ModelInfo(
                    name="llama3.2:3b",
                    size="3b",
                    parameters=3_000_000_000,
                    context_window=4096,
                    quantization="q4_0",
                ),
                "llama3.2:7b": ModelInfo(
                    name="llama3.2:7b",
                    size="7b",
                    parameters=7_000_000_000,
                    context_window=8192,
                    quantization="q4_0",
                ),
                "llama3.2:13b": ModelInfo(
                    name="llama3.2:13b",
                    size="13b",
                    parameters=13_000_000_000,
                    context_window=8192,
                    quantization="q4_0",
                ),
            }
        except Exception as e:
            print(f"Error refreshing model list: {e}")

    async def select_best_model(
        self, task_complexity: str = "medium", conversation_length: int = 0
    ) -> Tuple[str, float]:
        """Select the best model based on current conditions"""
        if not self.available_models:
            # Refresh if no models available
            await self._refresh_model_list()

        if not self.available_models:
            raise ValueError("No models available for selection")

        # Get current resources
        resources = self.resource_detector.get_current_resources()

        # Get performance degradation
        degradation = self.resource_detector.get_performance_degradation()

        # Filter models that can fit
        suitable_models = []
        for model_name, model_info in self.available_models.items():
            # Check if model fits in resources
            can_fit_result = self.resource_detector.can_fit_model(
                {"size": model_info.size, "parameters": model_info.parameters}
            )

            if can_fit_result["can_fit"]:
                # Calculate score based on capability and efficiency
                score = self._calculate_model_score(
                    model_info, resources, degradation, task_complexity, conversation_length
                )
                suitable_models.append((model_name, model_info, score))

        # Sort by score (descending)
        suitable_models.sort(key=lambda x: x[2], reverse=True)

        if not suitable_models:
            # No suitable models, return the smallest available as fallback
            smallest_model = min(self.available_models.items(), key=lambda x: x[1].parameters)
            return smallest_model[0], 0.5

        best_model_name, best_model_info, best_score = suitable_models[0]
        return best_model_name, best_score

    def _calculate_model_score(
        self,
        model_info: ModelInfo,
        resources: Any,
        degradation: Dict,
        task_complexity: str,
        conversation_length: int,
    ) -> float:
        """Calculate score for model selection"""
        score = 0.0

        # Base score from model capability (size)
        capability_scores = {
            ModelSize.TINY: 0.3,
            ModelSize.SMALL: 0.5,
            ModelSize.MEDIUM: 0.7,
            ModelSize.LARGE: 0.85,
            ModelSize.HUGE: 1.0,
        }
        score += capability_scores.get(model_info.size_category, 0.7)

        # Resource fit bonus
        if hasattr(model_info, "size_category"):
            resource_fit = self.resource_detector.can_fit_model(
                {"size": model_info.size_category.value, "parameters": model_info.parameters}
            )
            score += resource_fit["confidence"] * 0.3

        # Performance degradation penalty
        if degradation["overall"] == "critical":
            score -= 0.3
        elif degradation["overall"] == "degrading":
            score -= 0.15

        # Task complexity adjustment
        complexity_multipliers = {
            "simple": {"tiny": 1.2, "small": 1.1, "medium": 0.9, "large": 0.7, "huge": 0.5},
            "medium": {"tiny": 0.8, "small": 0.9, "medium": 1.0, "large": 1.1, "huge": 0.9},
            "complex": {"tiny": 0.5, "small": 0.7, "medium": 0.9, "large": 1.2, "huge": 1.3},
        }

        size_key = (
            model_info.size_category.value if hasattr(model_info, "size_category") else "medium"
        )
        mult = complexity_multipliers.get(task_complexity, {}).get(size_key, 1.0)
        score *= mult

        # Conversation length adjustment (larger context for longer conversations)
        if conversation_length > 50:
            if model_info.context_window >= 8192:
                score += 0.1
            elif model_info.context_window < 4096:
                score -= 0.2
        elif conversation_length > 20:
            if model_info.context_window >= 4096:
                score += 0.05

        return max(0.0, min(1.0, score))

    def should_switch_model(self, current_performance_metrics: Dict) -> SwitchRecommendation:
        """Determine if model should be switched"""
        if not self.current_model:
            # No current model, select best available
            return SwitchRecommendation(
                should_switch=True,
                target_model=None,  # Will be selected
                reason=SwitchReason.PROACTIVE_OPTIMIZATION,
                confidence=1.0,
                expected_benefit="Initialize with optimal model",
                estimated_cost={"time": 0.0, "memory": 0.0},
            )

        # Check resource constraints
        memory_constrained, constraint_level = self.resource_detector.is_memory_constrained()
        if memory_constrained and constraint_level in ["warning", "critical"]:
            # Need to switch to smaller model
            smaller_model = self._find_smaller_model()
            if smaller_model:
                benefit = f"Reduce memory usage during {constraint_level} constraint"
                return SwitchRecommendation(
                    should_switch=True,
                    target_model=smaller_model,
                    reason=SwitchReason.RESOURCE_CONSTRAINT,
                    confidence=0.9,
                    expected_benefit=benefit,
                    estimated_cost={"time": 2.0, "memory": -4.0},
                )

        # Check performance degradation
        degradation = self.resource_detector.get_performance_degradation()
        if degradation["overall"] in ["critical", "degrading"]:
            # Consider switching to smaller model
            smaller_model = self._find_smaller_model()
            if smaller_model and self.current_model_info:
                benefit = "Improve responsiveness during performance degradation"
                return SwitchRecommendation(
                    should_switch=True,
                    target_model=smaller_model,
                    reason=SwitchReason.PERFORMANCE_DEGRADATION,
                    confidence=0.8,
                    expected_benefit=benefit,
                    estimated_cost={"time": 2.0, "memory": -4.0},
                )

        # Check if resources are available for larger model
        if not memory_constrained and degradation["overall"] == "stable":
            # Can we switch to a larger model?
            larger_model = self._find_larger_model()
            if larger_model:
                benefit = "Increase capability with available resources"
                return SwitchRecommendation(
                    should_switch=True,
                    target_model=larger_model,
                    reason=SwitchReason.PROACTIVE_OPTIMIZATION,
                    confidence=0.7,
                    expected_benefit=benefit,
                    estimated_cost={"time": 3.0, "memory": 4.0},
                )

        return SwitchRecommendation(
            should_switch=False,
            target_model=None,
            reason=SwitchReason.PROACTIVE_OPTIMIZATION,
            confidence=1.0,
            expected_benefit="Current model is optimal",
            estimated_cost={"time": 0.0, "memory": 0.0},
        )

    def _find_smaller_model(self) -> Optional[str]:
        """Find a smaller model than current"""
        if not self.current_model_info or not self.available_models:
            return None

        current_size = getattr(self.current_model_info, "size_category", ModelSize.MEDIUM)
        smaller_sizes = [
            ModelSize.TINY,
            ModelSize.SMALL,
            ModelSize.MEDIUM,
            ModelSize.LARGE,
            ModelSize.HUGE,
        ]
        current_index = smaller_sizes.index(current_size)

        # Look for models in smaller categories
        for size in smaller_sizes[:current_index]:
            for model_name, model_info in self.available_models.items():
                if hasattr(model_info, "size_category") and model_info.size_category == size:
                    # Check if it fits
                    can_fit = self.resource_detector.can_fit_model(
                        {"size": size.value, "parameters": model_info.parameters}
                    )
                    if can_fit["can_fit"]:
                        return model_name

        return None

    def _find_larger_model(self) -> Optional[str]:
        """Find a larger model than current"""
        if not self.current_model_info or not self.available_models:
            return None

        current_size = getattr(self.current_model_info, "size_category", ModelSize.MEDIUM)
        larger_sizes = [
            ModelSize.TINY,
            ModelSize.SMALL,
            ModelSize.MEDIUM,
            ModelSize.LARGE,
            ModelSize.HUGE,
        ]
        current_index = larger_sizes.index(current_size)

        # Look for models in larger categories
        for size in larger_sizes[current_index + 1 :]:
            for model_name, model_info in self.available_models.items():
                if hasattr(model_info, "size_category") and model_info.size_category == size:
                    # Check if it fits
                    can_fit = self.resource_detector.can_fit_model(
                        {"size": size.value, "parameters": model_info.parameters}
                    )
                    if can_fit["can_fit"]:
                        return model_name

        return None

    async def switch_model(
        self, new_model_name: str, conversation_context: Optional[List[Dict]] = None
    ) -> SwitchMetrics:
        """Switch to a new model with context preservation"""
        start_time = time.time()

        try:
            # Validate new model is available
            if new_model_name not in self.available_models:
                raise ValueError(f"Model {new_model_name} not available")

            # Compress conversation context if provided
            context_transfer_time = 0.0
            compression_ratio = 1.0
            compressed_context = conversation_context

            if conversation_context:
                compress_start = time.time()
                compressed_context = self._compress_context(conversation_context)
                context_transfer_time = time.time() - compress_start
                compression_ratio = len(conversation_context) / max(1, len(compressed_context))

            # Perform the switch (mock implementation)
            # In real implementation, this would use the ollama client
            old_model = self.current_model
            self.current_model = new_model_name
            self.current_model_info = self.available_models[new_model_name]

            if conversation_context and compressed_context is not None:
                self.conversation_context = compressed_context

            switch_time = time.time() - start_time

            # Update performance metrics
            self._update_switch_metrics(True, switch_time)

            # Record switch in history
            self.switch_history.append(
                {
                    "timestamp": time.time(),
                    "from_model": old_model,
                    "to_model": new_model_name,
                    "switch_time": switch_time,
                    "context_transfer_time": context_transfer_time,
                    "compression_ratio": compression_ratio,
                    "success": True,
                }
            )

            return SwitchMetrics(
                switch_time=switch_time,
                context_transfer_time=context_transfer_time,
                context_compression_ratio=compression_ratio,
                success=True,
            )

        except Exception as e:
            switch_time = time.time() - start_time
            self._update_switch_metrics(False, switch_time)

            return SwitchMetrics(
                switch_time=switch_time,
                context_transfer_time=0.0,
                context_compression_ratio=1.0,
                success=False,
                error_message=str(e),
            )

    def _compress_context(self, context: List[Dict]) -> List[Dict]:
        """Compress conversation context for transfer"""
        # Simple compression strategy - keep recent messages and summaries
        if len(context) <= 10:
            return context

        # Keep first 2 and last 8 messages
        compressed = context[:2] + context[-8:]

        # Add a summary if we removed significant content
        if len(context) > len(compressed):
            summary_msg = {
                "role": "system",
                "content": f"[{len(context) - len(compressed)} earlier messages summarized for context compression]",
            }
            compressed.insert(2, summary_msg)

        return compressed

    def _update_switch_metrics(self, success: bool, switch_time: float):
        """Update performance metrics for switching"""
        self.performance_metrics["switch_count"] += 1

        if success:
            self.performance_metrics["successful_switches"] += 1

        # Update average switch time
        if self.performance_metrics["switch_count"] == 1:
            self.performance_metrics["average_switch_time"] = switch_time
        else:
            current_avg = self.performance_metrics["average_switch_time"]
            n = self.performance_metrics["switch_count"]
            new_avg = ((n - 1) * current_avg + switch_time) / n
            self.performance_metrics["average_switch_time"] = new_avg

        self.performance_metrics["last_switch_time"] = time.time()

    def get_model_recommendations(self) -> List[Dict]:
        """Get model recommendations based on current state"""
        recommendations = []

        # Get current resources
        resources = self.resource_detector.get_current_resources()

        # Get performance degradation
        degradation = self.resource_detector.get_performance_degradation()

        for model_name, model_info in self.available_models.items():
            # Check if model fits
            can_fit_result = self.resource_detector.can_fit_model(
                {"size": model_info.size, "parameters": model_info.parameters}
            )

            if can_fit_result["can_fit"]:
                # Calculate recommendation score
                score = self._calculate_model_score(model_info, resources, degradation, "medium", 0)

                recommendation = {
                    "model": model_name,
                    "model_info": asdict(model_info),
                    "can_fit": True,
                    "fit_confidence": can_fit_result["confidence"],
                    "performance_score": score,
                    "memory_deficit_gb": can_fit_result.get("memory_deficit_gb", 0),
                    "recommendation": can_fit_result.get("recommendation", ""),
                    "reason": self._get_recommendation_reason(score, resources, model_info),
                }
                recommendations.append(recommendation)
            else:
                # Model doesn't fit, but include with explanation
                recommendation = {
                    "model": model_name,
                    "model_info": asdict(model_info),
                    "can_fit": False,
                    "fit_confidence": can_fit_result["confidence"],
                    "performance_score": 0.0,
                    "memory_deficit_gb": can_fit_result.get("memory_deficit_gb", 0),
                    "recommendation": can_fit_result.get("recommendation", ""),
                    "reason": f"Insufficient memory - need {can_fit_result.get('memory_deficit_gb', 0):.1f}GB more",
                }
                recommendations.append(recommendation)

        # Sort by performance score
        recommendations.sort(key=lambda x: x["performance_score"], reverse=True)

        return recommendations

    def _get_recommendation_reason(
        self, score: float, resources: Any, model_info: ModelInfo
    ) -> str:
        """Get reason for recommendation"""
        if score >= 0.8:
            return "Excellent fit for current conditions"
        elif score >= 0.6:
            return "Good choice, should work well"
        elif score >= 0.4:
            return "Possible fit, may have performance issues"
        else:
            return "Not recommended for current conditions"

    def estimate_switching_cost(
        self, from_model: str, to_model: str, context_size: int
    ) -> Dict[str, float]:
        """Estimate the cost of switching between models"""
        # Base time cost
        base_time = 1.0  # 1 second base switching time

        # Context transfer cost
        context_time = context_size * 0.01  # 10ms per message

        # Model loading cost (based on size difference)
        from_info = self.available_models.get(from_model)
        to_info = self.available_models.get(to_model)

        if from_info and to_info:
            size_diff = abs(to_info.parameters - from_info.parameters) / 1e9
            loading_cost = size_diff * 0.5  # 0.5s per billion parameters difference
        else:
            loading_cost = 2.0  # Default 2 seconds

        total_time = base_time + context_time + loading_cost

        # Memory cost (temporary increase during switch)
        memory_cost = 2.0 if context_size > 20 else 1.0  # GB

        return {
            "time_seconds": total_time,
            "memory_gb": memory_cost,
            "context_transfer_time": context_time,
            "model_loading_time": loading_cost,
        }

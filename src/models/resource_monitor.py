"""System resource monitoring for intelligent model selection."""

import psutil
import time
from typing import Dict, List, Optional, Tuple
import logging


class ResourceMonitor:
    """Monitor system resources for model selection decisions."""

    def __init__(self, memory_threshold: float = 80.0, cpu_threshold: float = 80.0):
        """Initialize resource monitor.

        Args:
            memory_threshold: Memory usage % that triggers model switching
            cpu_threshold: CPU usage % that triggers model switching
        """
        self.memory_threshold = memory_threshold
        self.cpu_threshold = cpu_threshold
        self.logger = logging.getLogger(__name__)

        # Track resource history for trend analysis
        self.resource_history: List[Dict[str, float]] = []
        self.max_history_size = 100  # Keep last 100 samples

    def get_current_resources(self) -> Dict[str, float]:
        """Get current system resource usage.

        Returns:
            Dict with:
            - memory_percent: Memory usage percentage (0-100)
            - cpu_percent: CPU usage percentage (0-100)
            - available_memory_gb: Available RAM in GB
            - gpu_vram_gb: Available GPU VRAM in GB (0 if no GPU)
        """
        try:
            # Memory information
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            available_memory_gb = memory.available / (1024**3)

            # CPU information
            cpu_percent = psutil.cpu_percent(interval=1)

            # GPU information (if available)
            gpu_vram_gb = self._get_gpu_memory()

            return {
                "memory_percent": memory_percent,
                "cpu_percent": cpu_percent,
                "available_memory_gb": available_memory_gb,
                "gpu_vram_gb": gpu_vram_gb,
            }

        except Exception as e:
            self.logger.error(f"Failed to get system resources: {e}")
            return {
                "memory_percent": 0.0,
                "cpu_percent": 0.0,
                "available_memory_gb": 0.0,
                "gpu_vram_gb": 0.0,
            }

    def get_resource_trend(self, window_minutes: int = 5) -> Dict[str, str]:
        """Analyze resource usage trend over time window.

        Args:
            window_minutes: Time window in minutes to analyze

        Returns:
            Dict with trend indicators: "increasing", "decreasing", "stable"
        """
        cutoff_time = time.time() - (window_minutes * 60)

        # Filter recent history
        recent_data = [
            entry
            for entry in self.resource_history
            if entry.get("timestamp", 0) > cutoff_time
        ]

        if len(recent_data) < 2:
            return {"memory": "insufficient_data", "cpu": "insufficient_data"}

        # Calculate trends
        memory_trend = self._calculate_trend([entry["memory"] for entry in recent_data])
        cpu_trend = self._calculate_trend([entry["cpu"] for entry in recent_data])

        return {
            "memory": memory_trend,
            "cpu": cpu_trend,
        }

    def can_load_model(self, model_size_gb: float) -> bool:
        """Check if enough resources are available to load a model.

        Args:
            model_size_gb: Required memory in GB for the model

        Returns:
            True if model can be loaded, False otherwise
        """
        resources = self.get_current_resources()

        # Check if enough available memory (with 50% safety margin)
        required_memory_with_margin = model_size_gb * 1.5
        available_memory = resources["available_memory_gb"]

        if available_memory < required_memory_with_margin:
            self.logger.warning(
                f"Insufficient memory: need {required_memory_with_margin:.1f}GB, "
                f"have {available_memory:.1f}GB"
            )
            return False

        # Check if GPU has enough VRAM if available
        if resources["gpu_vram_gb"] > 0:
            if resources["gpu_vram_gb"] < model_size_gb:
                self.logger.warning(
                    f"Insufficient GPU VRAM: need {model_size_gb:.1f}GB, "
                    f"have {resources['gpu_vram_gb']:.1f}GB"
                )
                return False

        return True

    def is_system_overloaded(self) -> bool:
        """Check if system resources exceed configured thresholds.

        Returns:
            True if system is overloaded, False otherwise
        """
        resources = self.get_current_resources()

        # Check memory threshold
        if resources["memory_percent"] > self.memory_threshold:
            return True

        # Check CPU threshold
        if resources["cpu_percent"] > self.cpu_threshold:
            return True

        return False

    def update_history(self) -> None:
        """Update resource history for trend analysis."""
        resources = self.get_current_resources()

        # Add timestamp and sample
        resources["timestamp"] = time.time()
        self.resource_history.append(resources)

        # Trim history if too large
        if len(self.resource_history) > self.max_history_size:
            self.resource_history = self.resource_history[-self.max_history_size :]

    def get_best_model_size(self) -> str:
        """Recommend model size category based on current resources.

        Returns:
            Model size category: "small", "medium", or "large"
        """
        resources = self.get_current_resources()

        available_memory_gb = resources["available_memory_gb"]

        if available_memory_gb >= 8:
            return "large"
        elif available_memory_gb >= 4:
            return "medium"
        else:
            return "small"

    def _get_gpu_memory(self) -> float:
        """Get available GPU VRAM if GPU is available.

        Returns:
            Available GPU VRAM in GB, 0 if no GPU available
        """
        try:
            # Try to import gpu-tracker if available
            import gpu_tracker as gt

            # Get GPU information
            gpu_info = gt.get_gpus()

            # Get GPU information
            gpu_info = gt.get_gpus()
            if gpu_info:
                # Return available VRAM from first GPU
                total_vram = gpu_info[0].memory_total
                used_vram = gpu_info[0].memory_used
                available_vram = total_vram - used_vram
                return available_vram / 1024  # Convert MB to GB

        except ImportError:
            # gpu-tracker not installed, fall back to basic GPU detection
            pass
        except Exception as e:
            self.logger.debug(f"GPU tracking failed: {e}")

        return 0.0

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from a list of values.

        Args:
            values: List of numeric values in chronological order

        Returns:
            Trend indicator: "increasing", "decreasing", or "stable"
        """
        if len(values) < 2:
            return "insufficient_data"

        # Simple linear regression to determine trend
        n = len(values)
        x_values = list(range(n))

        # Calculate slope
        sum_x = sum(x_values)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(x_values, values))
        sum_x2 = sum(x * x for x in x_values)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)

        # Determine trend based on slope magnitude
        if abs(slope) < 0.1:
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"

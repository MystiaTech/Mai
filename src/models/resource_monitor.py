"""System resource monitoring for intelligent model selection."""

import psutil
import time
from typing import Dict, List, Optional, Tuple
import logging

# Try to import pynvml for NVIDIA GPU monitoring
try:
    import pynvml

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    pynvml = None


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

        # Cache GPU info to avoid repeated initialization overhead
        self._gpu_cache: Optional[Dict[str, float]] = None
        self._gpu_cache_time: float = 0
        self._gpu_cache_duration: float = 1.0  # Cache for 1 second

        # Track if we've already tried pynvml and failed
        self._pynvml_failed: bool = False

    def get_current_resources(self) -> Dict[str, float]:
        """Get current system resource usage.

        Returns:
            Dict with:
            - memory_percent: Memory usage percentage (0-100)
            - cpu_percent: CPU usage percentage (0-100)
            - available_memory_gb: Available RAM in GB
            - gpu_vram_gb: Available GPU VRAM in GB (0 if no GPU)
            - gpu_total_vram_gb: Total VRAM capacity in GB (0 if no GPU)
            - gpu_used_vram_gb: Used VRAM in GB (0 if no GPU)
            - gpu_free_vram_gb: Available VRAM in GB (0 if no GPU)
            - gpu_utilization_percent: GPU utilization (0-100, 0 if no GPU)
            - gpu_temperature_c: GPU temperature in Celsius (0 if no GPU)
        """
        try:
            # Memory information
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            available_memory_gb = memory.available / (1024**3)

            # CPU information (use very short interval for performance)
            cpu_percent = psutil.cpu_percent(interval=0.05)

            # GPU information (if available) - with caching for performance
            gpu_info = self._get_cached_gpu_info()

            return {
                "memory_percent": memory_percent,
                "cpu_percent": cpu_percent,
                "available_memory_gb": available_memory_gb,
                "gpu_vram_gb": gpu_info.get(
                    "free_vram_gb", 0.0
                ),  # Backward compatibility
                "gpu_total_vram_gb": gpu_info.get("total_vram_gb", 0.0),
                "gpu_used_vram_gb": gpu_info.get("used_vram_gb", 0.0),
                "gpu_free_vram_gb": gpu_info.get("free_vram_gb", 0.0),
                "gpu_utilization_percent": gpu_info.get("utilization_percent", 0.0),
                "gpu_temperature_c": gpu_info.get("temperature_c", 0.0),
            }

        except Exception as e:
            self.logger.error(f"Failed to get system resources: {e}")
            return {
                "memory_percent": 0.0,
                "cpu_percent": 0.0,
                "available_memory_gb": 0.0,
                "gpu_vram_gb": 0.0,
                "gpu_total_vram_gb": 0.0,
                "gpu_used_vram_gb": 0.0,
                "gpu_free_vram_gb": 0.0,
                "gpu_utilization_percent": 0.0,
                "gpu_temperature_c": 0.0,
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

    def _get_cached_gpu_info(self) -> Dict[str, float]:
        """Get GPU info with caching to avoid repeated initialization overhead.

        Returns:
            GPU info dict (cached or fresh)
        """
        current_time = time.time()

        # Return cached info if still valid
        if (
            self._gpu_cache is not None
            and current_time - self._gpu_cache_time < self._gpu_cache_duration
        ):
            return self._gpu_cache

        # Get fresh GPU info and cache it
        self._gpu_cache = self._get_gpu_info()
        self._gpu_cache_time = current_time

        return self._gpu_cache

    def _get_gpu_info(self) -> Dict[str, float]:
        """Get detailed GPU information using pynvml or fallback methods.

        Returns:
            Dict with GPU metrics:
            - total_vram_gb: Total VRAM capacity in GB
            - used_vram_gb: Used VRAM in GB
            - free_vram_gb: Available VRAM in GB
            - utilization_percent: GPU utilization (0-100)
            - temperature_c: GPU temperature in Celsius
        """
        gpu_info = {
            "total_vram_gb": 0.0,
            "used_vram_gb": 0.0,
            "free_vram_gb": 0.0,
            "utilization_percent": 0.0,
            "temperature_c": 0.0,
        }

        # Try pynvml first for NVIDIA GPUs (but not if we already know it failed)
        if PYNVML_AVAILABLE and pynvml is not None and not self._pynvml_failed:
            try:
                # Initialize pynvml
                pynvml.nvmlInit()

                # Get number of GPUs
                device_count = pynvml.nvmlDeviceGetCount()
                if device_count > 0:
                    # Use first GPU (can be extended for multi-GPU support)
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

                    # Get memory info
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    total_bytes = memory_info.total
                    used_bytes = memory_info.used
                    free_bytes = memory_info.free

                    # Convert to GB
                    gpu_info["total_vram_gb"] = total_bytes / (1024**3)
                    gpu_info["used_vram_gb"] = used_bytes / (1024**3)
                    gpu_info["free_vram_gb"] = free_bytes / (1024**3)

                    # Get utilization (GPU and memory)
                    try:
                        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        gpu_info["utilization_percent"] = utilization.gpu
                    except Exception:
                        # Some GPUs don't support utilization queries
                        pass

                    # Get temperature
                    try:
                        temp = pynvml.nvmlDeviceGetTemperature(
                            handle, pynvml.NVML_TEMPERATURE_GPU
                        )
                        gpu_info["temperature_c"] = float(temp)
                    except Exception:
                        # Some GPUs don't support temperature queries
                        pass

                # Always shutdown pynvml when done
                pynvml.nvmlShutdown()

                self.logger.debug(
                    f"GPU detected via pynvml: {gpu_info['total_vram_gb']:.1f}GB total, "
                    f"{gpu_info['used_vram_gb']:.1f}GB used, "
                    f"{gpu_info['utilization_percent']:.0f}% utilization, "
                    f"{gpu_info['temperature_c']:.0f}°C"
                )
                return gpu_info

            except Exception as e:
                self.logger.debug(f"pynvml GPU detection failed: {e}")
                # Mark pynvml as failed to avoid repeated attempts
                self._pynvml_failed = True
                # Fall through to gpu-tracker

        # Fallback to gpu-tracker for other GPUs or when pynvml fails
        try:
            import gpu_tracker as gt

            gpu_list = gt.get_gpus()
            if gpu_list:
                gpu = gpu_list[0]  # Use first GPU

                # Convert MB to GB for consistency
                total_mb = getattr(gpu, "memory_total", 0)
                used_mb = getattr(gpu, "memory_used", 0)

                gpu_info["total_vram_gb"] = total_mb / 1024.0
                gpu_info["used_vram_gb"] = used_mb / 1024.0
                gpu_info["free_vram_gb"] = (total_mb - used_mb) / 1024.0

                self.logger.debug(
                    f"GPU detected via gpu-tracker: {gpu_info['total_vram_gb']:.1f}GB total, "
                    f"{gpu_info['used_vram_gb']:.1f}GB used"
                )
                return gpu_info

        except ImportError:
            self.logger.debug("gpu-tracker not available")
        except Exception as e:
            self.logger.debug(f"gpu-tracker failed: {e}")

        # No GPU detected - return default values
        self.logger.debug("No GPU detected")
        return gpu_info

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

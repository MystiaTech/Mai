"""
Resource monitoring for Mai.

Monitors system resources (CPU, RAM, GPU) and provides
resource-aware model selection capabilities.
"""

import time
import platform
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any
from collections import deque


@dataclass
class ResourceInfo:
    """Current system resource state"""

    cpu_percent: float
    memory_total_gb: float
    memory_available_gb: float
    memory_percent: float
    gpu_available: bool
    gpu_memory_gb: Optional[float] = None
    gpu_usage_percent: Optional[float] = None
    timestamp: float = 0.0


@dataclass
class MemoryTrend:
    """Memory usage trend analysis"""

    current: float
    trend: str  # 'stable', 'increasing', 'decreasing'
    rate: float  # GB per minute
    confidence: float  # 0.0 to 1.0


class ResourceDetector:
    """System resource monitoring with trend analysis"""

    def __init__(self):
        """Initialize resource monitoring"""
        self.memory_threshold_warning = 80.0  # 80% for warning
        self.memory_threshold_critical = 90.0  # 90% for critical
        self.history_window = 60  # seconds
        self.history_size = 60  # data points

        # Resource history tracking
        self.memory_history: deque = deque(maxlen=self.history_size)
        self.cpu_history: deque = deque(maxlen=self.history_size)
        self.timestamps: deque = deque(maxlen=self.history_size)

        # GPU detection
        self.gpu_available = self._detect_gpu()
        self.gpu_info = self._get_gpu_info()

        # Initialize psutil if available
        self._init_psutil()

    def _init_psutil(self):
        """Initialize psutil with fallback"""
        try:
            import psutil

            self.psutil = psutil
            self.has_psutil = True
        except ImportError:
            print("Warning: psutil not available. Resource monitoring will be limited.")
            self.psutil = None
            self.has_psutil = False

    def _detect_gpu(self) -> bool:
        """Detect GPU availability"""
        try:
            # Try NVIDIA GPU detection
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        try:
            # Try AMD GPU detection
            result = subprocess.run(
                ["rocm-smi", "--showproductname"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Apple Silicon detection
        if platform.system() == "Darwin" and platform.machine() in ["arm64", "arm"]:
            return True

        return False

    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information"""
        info: Dict[str, Any] = {"type": None, "memory_gb": None, "name": None}

        try:
            # NVIDIA GPU
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                if lines and lines[0]:
                    parts = lines[0].split(", ")
                    if len(parts) >= 2:
                        info["type"] = "nvidia"
                        info["name"] = parts[0].strip()
                        info["memory_gb"] = float(parts[1].strip()) / 1024  # Convert MB to GB
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pass

        # Apple Silicon
        if (
            not info["type"]
            and platform.system() == "Darwin"
            and platform.machine() in ["arm64", "arm"]
        ):
            info["type"] = "apple_silicon"
            # Unified memory, estimate based on system memory
            if self.has_psutil and self.psutil is not None:
                memory = self.psutil.virtual_memory()
                info["memory_gb"] = memory.total / (1024**3)

        return info

    def detect_resources(self) -> ResourceInfo:
        """Get current system resource state (alias for get_current_resources)"""
        return self.get_current_resources()

    def get_current_resources(self) -> ResourceInfo:
        """Get current system resource state"""
        if not self.has_psutil:
            # Fallback to basic monitoring
            return self._get_fallback_resources()

        # CPU usage
        cpu_percent = self.psutil.cpu_percent(interval=1) if self.psutil else 0.0

        # Memory information
        if self.psutil:
            memory = self.psutil.virtual_memory()
            memory_total_gb = memory.total / (1024**3)
            memory_available_gb = memory.available / (1024**3)
            memory_percent = memory.percent
        else:
            # Use fallback values
            memory_total_gb = 8.0  # Default assumption
            memory_available_gb = 4.0
            memory_percent = 50.0

        # GPU information
        gpu_usage_percent = None
        gpu_memory_gb = None

        if self.gpu_info["type"] == "nvidia":
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split("\n")
                    if lines and lines[0]:
                        parts = lines[0].split(", ")
                        if len(parts) >= 2:
                            gpu_usage_percent = float(parts[0].strip())
                            gpu_memory_gb = float(parts[1].strip()) / 1024
            except (subprocess.TimeoutExpired, ValueError):
                pass

        current_time = time.time()
        resource_info = ResourceInfo(
            cpu_percent=cpu_percent,
            memory_total_gb=memory_total_gb,
            memory_available_gb=memory_available_gb,
            memory_percent=memory_percent,
            gpu_available=self.gpu_available,
            gpu_memory_gb=gpu_memory_gb,
            gpu_usage_percent=gpu_usage_percent,
            timestamp=current_time,
        )

        # Update history
        self._update_history(resource_info)

        return resource_info

    def _get_fallback_resources(self) -> ResourceInfo:
        """Fallback resource detection without psutil"""
        # Basic resource detection using /proc filesystem on Linux
        cpu_percent = 0.0
        memory_total_gb = 0.0
        memory_available_gb = 0.0
        memory_percent = 0.0

        try:
            # Read memory info from /proc/meminfo
            with open("/proc/meminfo", "r") as f:
                meminfo = {}
                for line in f:
                    if ":" in line:
                        key, value = line.split(":", 1)
                        meminfo[key.strip()] = int(value.split()[0])

                if "MemTotal" in meminfo:
                    memory_total_gb = meminfo["MemTotal"] / (1024**2)
                if "MemAvailable" in meminfo:
                    memory_available_gb = meminfo["MemAvailable"] / (1024**2)

                if memory_total_gb > 0:
                    memory_percent = (
                        (memory_total_gb - memory_available_gb) / memory_total_gb
                    ) * 100
        except (IOError, KeyError, ValueError):
            pass

        current_time = time.time()
        return ResourceInfo(
            cpu_percent=cpu_percent,
            memory_total_gb=memory_total_gb,
            memory_available_gb=memory_available_gb,
            memory_percent=memory_percent,
            gpu_available=self.gpu_available,
            gpu_memory_gb=self.gpu_info.get("memory_gb"),
            gpu_usage_percent=None,
            timestamp=current_time,
        )

    def _update_history(self, resource_info: ResourceInfo):
        """Update resource history for trend analysis"""
        current_time = time.time()

        self.memory_history.append(resource_info.memory_percent)
        self.cpu_history.append(resource_info.cpu_percent)
        self.timestamps.append(current_time)

    def is_memory_constrained(self) -> Tuple[bool, str]:
        """Check if memory is constrained"""
        if not self.memory_history:
            resources = self.get_current_resources()
            current_memory = resources.memory_percent
        else:
            current_memory = self.memory_history[-1]

        # Check current memory usage
        if current_memory >= self.memory_threshold_critical:
            return True, "critical"
        elif current_memory >= self.memory_threshold_warning:
            return True, "warning"

        # Check trend
        trend = self.get_memory_trend()
        if trend.trend == "increasing" and trend.rate > 5.0:  # 5GB/min increase
            return True, "trend_warning"

        return False, "normal"

    def get_memory_trend(self) -> MemoryTrend:
        """Analyze memory usage trend over last minute"""
        if len(self.memory_history) < 10:
            return MemoryTrend(
                current=self.memory_history[-1] if self.memory_history else 0.0,
                trend="stable",
                rate=0.0,
                confidence=0.0,
            )

        # Get recent data points (last 10 measurements)
        recent_memory = list(self.memory_history)[-10:]
        recent_times = list(self.timestamps)[-10:]

        # Calculate trend
        if len(recent_memory) >= 2 and len(recent_times) >= 2:
            time_span = recent_times[-1] - recent_times[0]
            memory_change = recent_memory[-1] - recent_memory[0]

            # Convert to GB per minute if we have memory info
            rate = 0.0
            if self.has_psutil and time_span > 0 and self.psutil is not None:
                # Use psutil to get total memory for conversion
                total_memory = self.psutil.virtual_memory().total / (1024**3)
                rate = (memory_change / 100.0) * total_memory * (60.0 / time_span)

            # Determine trend
            if abs(memory_change) < 2.0:  # Less than 2% change
                trend = "stable"
            elif memory_change > 0:
                trend = "increasing"
            else:
                trend = "decreasing"

            # Confidence based on data consistency
            confidence = min(1.0, len(recent_memory) / 10.0)

            return MemoryTrend(
                current=recent_memory[-1], trend=trend, rate=rate, confidence=confidence
            )

        return MemoryTrend(
            current=recent_memory[-1] if recent_memory else 0.0,
            trend="stable",
            rate=0.0,
            confidence=0.0,
        )

    def get_performance_degradation(self) -> Dict:
        """Analyze performance degradation metrics"""
        if len(self.memory_history) < 20 or len(self.cpu_history) < 20:
            return {
                "status": "insufficient_data",
                "memory_trend": "unknown",
                "cpu_trend": "unknown",
                "overall": "stable",
            }

        # Memory trend
        memory_trend = self.get_memory_trend()

        # CPU trend
        recent_cpu = list(self.cpu_history)[-10:]
        older_cpu = list(self.cpu_history)[-20:-10]

        avg_recent_cpu = sum(recent_cpu) / len(recent_cpu)
        avg_older_cpu = sum(older_cpu) / len(older_cpu)

        cpu_increase = avg_recent_cpu - avg_older_cpu

        # Overall assessment
        if memory_trend.trend == "increasing" and memory_trend.rate > 5.0:
            memory_status = "worsening"
        elif memory_trend.trend == "increasing":
            memory_status = "concerning"
        else:
            memory_status = "stable"

        if cpu_increase > 20:
            cpu_status = "worsening"
        elif cpu_increase > 10:
            cpu_status = "concerning"
        else:
            cpu_status = "stable"

        # Overall status
        if memory_status == "worsening" or cpu_status == "worsening":
            overall = "critical"
        elif memory_status == "concerning" or cpu_status == "concerning":
            overall = "degrading"
        else:
            overall = "stable"

        return {
            "status": "analyzed",
            "memory_trend": memory_status,
            "cpu_trend": cpu_status,
            "cpu_increase": cpu_increase,
            "memory_rate": memory_trend.rate,
            "overall": overall,
        }

    def estimate_model_requirements(self, model_size: str) -> Dict:
        """Estimate memory requirements for model size"""
        # Conservative estimates based on model parameter count
        requirements = {
            "1b": {"memory_gb": 2.0, "memory_warning_gb": 2.5, "memory_critical_gb": 3.0},
            "3b": {"memory_gb": 4.0, "memory_warning_gb": 5.0, "memory_critical_gb": 6.0},
            "7b": {"memory_gb": 8.0, "memory_warning_gb": 10.0, "memory_critical_gb": 12.0},
            "13b": {"memory_gb": 16.0, "memory_warning_gb": 20.0, "memory_critical_gb": 24.0},
            "70b": {"memory_gb": 80.0, "memory_warning_gb": 100.0, "memory_critical_gb": 120.0},
        }

        size_key = model_size.lower()
        if size_key not in requirements:
            # Default to 7B requirements for unknown models
            size_key = "7b"

        base_req = requirements[size_key]

        # Add buffer for context and processing overhead (50%)
        context_overhead = base_req["memory_gb"] * 0.5

        return {
            "size_category": size_key,
            "base_memory_gb": base_req["memory_gb"],
            "context_overhead_gb": context_overhead,
            "total_required_gb": base_req["memory_gb"] + context_overhead,
            "warning_threshold_gb": base_req["memory_warning_gb"],
            "critical_threshold_gb": base_req["memory_critical_gb"],
        }

    def can_fit_model(self, model_info: Dict) -> Dict:
        """Check if model fits in current resources"""
        # Extract model size info
        model_size = model_info.get("size", "7b")
        if isinstance(model_size, str):
            # Extract numeric size from strings like "7B", "13B", etc.
            import re

            match = re.search(r"(\d+\.?\d*)[Bb]", model_size)
            if match:
                size_num = float(match.group(1))
                if size_num <= 2:
                    size_key = "1b"
                elif size_num <= 4:
                    size_key = "3b"
                elif size_num <= 10:
                    size_key = "7b"
                elif size_num <= 20:
                    size_key = "13b"
                else:
                    size_key = "70b"
            else:
                size_key = "7b"
        else:
            size_key = str(model_size).lower()

        # Get requirements
        requirements = self.estimate_model_requirements(size_key)

        # Get current resources
        current_resources = self.get_current_resources()

        # Check memory fit
        available_memory = current_resources.memory_available_gb
        required_memory = requirements["total_required_gb"]

        memory_fit_score = min(1.0, available_memory / required_memory)

        # Check performance trends
        degradation = self.get_performance_degradation()

        # Adjust confidence based on trends
        trend_adjustment = 1.0
        if degradation["overall"] == "critical":
            trend_adjustment = 0.5
        elif degradation["overall"] == "degrading":
            trend_adjustment = 0.8

        confidence = memory_fit_score * trend_adjustment

        # GPU consideration
        gpu_factor = 1.0
        if self.gpu_available and self.gpu_info.get("memory_gb"):
            gpu_memory = self.gpu_info["memory_gb"]
            if gpu_memory < required_memory:
                gpu_factor = 0.5  # GPU might not have enough memory

        final_confidence = confidence * gpu_factor

        return {
            "can_fit": final_confidence >= 0.8,
            "confidence": final_confidence,
            "memory_fit_score": memory_fit_score,
            "trend_adjustment": trend_adjustment,
            "gpu_factor": gpu_factor,
            "available_memory_gb": available_memory,
            "required_memory_gb": required_memory,
            "memory_deficit_gb": max(0, required_memory - available_memory),
            "recommendation": self._get_fitting_recommendation(final_confidence, requirements),
        }

    def _get_fitting_recommendation(self, confidence: float, requirements: Dict) -> str:
        """Get recommendation based on fitting assessment"""
        if confidence >= 0.9:
            return "Excellent fit - model should run smoothly"
        elif confidence >= 0.8:
            return "Good fit - model should work well"
        elif confidence >= 0.6:
            return "Possible fit - may experience performance issues"
        elif confidence >= 0.4:
            return "Tight fit - expect significant slowdowns"
        else:
            return f"Insufficient resources - need at least {requirements['total_required_gb']:.1f}GB available"


# Required import for subprocess
import subprocess

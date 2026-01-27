"""
Resource Enforcement for Mai Sandbox System

Provides percentage-based resource limit enforcement
building on existing Phase 1 monitoring infrastructure.
"""

import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ResourceLimits:
    """Resource limit configuration"""

    cpu_percent: float
    memory_percent: float
    timeout_seconds: int
    network_bandwidth_mbps: float | None = None


@dataclass
class ResourceUsage:
    """Current resource usage statistics"""

    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    elapsed_seconds: float
    approaching_limits: dict[str, bool]


class ResourceEnforcer:
    """
    Enforces resource limits for sandbox execution.
    Builds on Phase 1 ResourceDetector for percentage-based limits.
    """

    def __init__(self):
        """Initialize resource enforcer"""
        # Try to import existing resource monitoring from Phase 1
        try:
            sys.path.append(str(Path(__file__).parent.parent / "model"))
            from resource_detector import ResourceDetector

            self.resource_detector = ResourceDetector()
        except ImportError:
            # Fallback implementation
            self.resource_detector = None

        self.current_limits: ResourceLimits | None = None
        self.start_time: float | None = None
        self.timeout_timer: threading.Timer | None = None
        self.monitoring_active: bool = False

    def set_cpu_limit(self, percent: float) -> float:
        """
        Calculate CPU limit as percentage of available resources

        Args:
            percent: Desired CPU limit (0-100)

        Returns:
            Actual CPU limit percentage
        """
        if not 0 <= percent <= 100:
            raise ValueError("CPU percent must be between 0 and 100")

        # Calculate effective limit
        cpu_limit = min(percent, 100.0)

        return cpu_limit

    def set_memory_limit(self, percent: float) -> float:
        """
        Calculate memory limit as percentage of available resources

        Args:
            percent: Desired memory limit (0-100)

        Returns:
            Actual memory limit percentage
        """
        if not 0 <= percent <= 100:
            raise ValueError("Memory percent must be between 0 and 100")

        # Calculate effective limit
        if self.resource_detector:
            try:
                resource_info = self.resource_detector.get_current_usage()
                memory_limit = min(
                    percent,
                    resource_info.memory_percent
                    + (resource_info.memory_available_gb / resource_info.memory_total_gb * 100),
                )
                return memory_limit
            except Exception:
                pass

        # Fallback
        memory_limit = min(percent, 100.0)
        return memory_limit

    def set_limits(self, limits: ResourceLimits) -> bool:
        """
        Set comprehensive resource limits

        Args:
            limits: ResourceLimits configuration

        Returns:
            True if limits were successfully set
        """
        try:
            self.current_limits = limits
            return True
        except Exception as e:
            print(f"Failed to set limits: {e}")
            return False

    def enforce_timeout(self, seconds: int) -> bool:
        """
        Enforce execution timeout using signal alarm

        Args:
            seconds: Timeout in seconds

        Returns:
            True if timeout was set successfully
        """
        try:
            if self.timeout_timer:
                self.timeout_timer.cancel()

            # Create timeout handler
            def timeout_handler():
                raise TimeoutError(f"Execution exceeded {seconds} second timeout")

            # Set timer (cross-platform alternative to signal.alarm)
            self.timeout_timer = threading.Timer(seconds, timeout_handler)
            self.timeout_timer.daemon = True
            self.timeout_timer.start()

            return True
        except Exception as e:
            print(f"Failed to set timeout: {e}")
            return False

    def start_monitoring(self) -> bool:
        """
        Start resource monitoring for an execution session

        Returns:
            True if monitoring started successfully
        """
        try:
            self.start_time = time.time()
            self.monitoring_active = True
            return True
        except Exception as e:
            print(f"Failed to start monitoring: {e}")
            return False

    def stop_monitoring(self) -> ResourceUsage:
        """
        Stop monitoring and return usage statistics

        Returns:
            ResourceUsage with execution statistics
        """
        if not self.monitoring_active:
            raise RuntimeError("Monitoring not active")

        # Stop timeout timer
        if self.timeout_timer:
            self.timeout_timer.cancel()
            self.timeout_timer = None

        # Calculate usage
        end_time = time.time()
        elapsed = end_time - (self.start_time or 0)

        # Get current resource info
        cpu_percent = 0.0
        memory_percent = 0.0
        memory_used_gb = 0.0
        memory_total_gb = 0.0

        if self.resource_detector:
            try:
                current_info = self.resource_detector.get_current_usage()
                cpu_percent = current_info.cpu_percent
                memory_percent = current_info.memory_percent
                memory_used_gb = current_info.memory_total_gb - current_info.memory_available_gb
            except Exception:
                pass  # Use fallback values

        # Check approaching limits
        approaching = {}
        if self.current_limits:
            approaching["cpu"] = cpu_percent > self.current_limits.cpu_percent * 0.8
            approaching["memory"] = memory_percent > self.current_limits.memory_percent * 0.8
            approaching["timeout"] = elapsed > self.current_limits.timeout_seconds * 0.8

        usage = ResourceUsage(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_gb=memory_used_gb,
            elapsed_seconds=elapsed,
            approaching_limits=approaching,
        )

        self.monitoring_active = False
        return usage

    def monitor_usage(self) -> dict[str, Any]:
        """
        Get current resource usage statistics

        Returns:
            Dictionary with current usage metrics
        """
        # Get current resource info
        cpu_percent = 0.0
        memory_percent = 0.0
        memory_used_gb = 0.0
        memory_available_gb = 0.0
        memory_total_gb = 0.0
        gpu_available = False
        gpu_memory_gb = None
        gpu_usage_percent = None

        if self.resource_detector:
            try:
                current_info = self.resource_detector.get_current_usage()
                cpu_percent = current_info.cpu_percent
                memory_percent = current_info.memory_percent
                memory_used_gb = current_info.memory_total_gb - current_info.memory_available_gb
                memory_available_gb = current_info.memory_available_gb
                memory_total_gb = current_info.memory_total_gb
                gpu_available = current_info.gpu_available
                gpu_memory_gb = current_info.gpu_memory_gb
                gpu_usage_percent = current_info.gpu_usage_percent
            except Exception:
                pass

        usage = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "memory_used_gb": memory_used_gb,
            "memory_available_gb": memory_available_gb,
            "memory_total_gb": memory_total_gb,
            "gpu_available": gpu_available,
            "gpu_memory_gb": gpu_memory_gb,
            "gpu_usage_percent": gpu_usage_percent,
            "monitoring_active": self.monitoring_active,
        }

        if self.monitoring_active and self.start_time:
            usage["elapsed_seconds"] = time.time() - self.start_time

        return usage

    def check_limits(self) -> dict[str, bool]:
        """
        Check if current usage exceeds or approaches limits

        Returns:
            Dictionary of limit check results
        """
        if not self.current_limits:
            return {"limits_set": False}

        # Get current resource info
        cpu_percent = 0.0
        memory_percent = 0.0

        if self.resource_detector:
            try:
                current_info = self.resource_detector.get_current_usage()
                cpu_percent = current_info.cpu_percent
                memory_percent = current_info.memory_percent
            except Exception:
                pass

        checks = {
            "limits_set": True,
            "cpu_exceeded": cpu_percent > self.current_limits.cpu_percent,
            "memory_exceeded": memory_percent > self.current_limits.memory_percent,
            "cpu_approaching": cpu_percent > self.current_limits.cpu_percent * 0.8,
            "memory_approaching": memory_percent > self.current_limits.memory_percent * 0.8,
        }

        if self.monitoring_active and self.start_time:
            elapsed = time.time() - self.start_time
            checks["timeout_exceeded"] = elapsed > self.current_limits.timeout_seconds
            checks["timeout_approaching"] = elapsed > self.current_limits.timeout_seconds * 0.8

        return checks

    def graceful_degradation_warning(self) -> str | None:
        """
        Generate warning if approaching resource limits

        Returns:
            Warning message or None if safe
        """
        checks = self.check_limits()

        if not checks["limits_set"]:
            return None

        warnings = []

        if checks["cpu_approaching"]:
            warnings.append(f"CPU usage approaching limit ({self.current_limits.cpu_percent}%)")

        if checks["memory_approaching"]:
            warnings.append(
                f"Memory usage approaching limit ({self.current_limits.memory_percent}%)"
            )

        if self.monitoring_active and self.start_time:
            elapsed = time.time() - self.start_time
            if elapsed > self.current_limits.timeout_seconds * 0.8:
                warnings.append(
                    f"Execution approaching timeout ({self.current_limits.timeout_seconds}s)"
                )

        if warnings:
            return "Warning: " + "; ".join(warnings) + ". Consider reducing execution scope."

        return None

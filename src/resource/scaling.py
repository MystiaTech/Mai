"""Proactive scaling system with hybrid monitoring and graceful degradation."""

import asyncio
import threading
import time
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import deque

from .tiers import HardwareTierDetector
from ..models.resource_monitor import ResourceMonitor


class ScalingDecision(Enum):
    """Types of scaling decisions."""

    NO_CHANGE = "no_change"
    UPGRADE = "upgrade"
    DOWNGRADE = "downgrade"
    DEGRADATION_CASCADE = "degradation_cascade"


@dataclass
class ScalingEvent:
    """Record of a scaling decision and its context."""

    timestamp: float
    decision: ScalingDecision
    old_model_size: Optional[str]
    new_model_size: Optional[str]
    reason: str
    resources: Dict[str, float]
    tier: str


class ProactiveScaler:
    """
    Proactive scaling system with hybrid monitoring and graceful degradation.

    Combines continuous background monitoring with pre-flight checks to
    anticipate resource constraints and scale models before performance
    degradation impacts user experience.
    """

    def __init__(
        self,
        resource_monitor: Optional[ResourceMonitor] = None,
        tier_detector: Optional[HardwareTierDetector] = None,
        upgrade_threshold: float = 0.8,
        downgrade_threshold: float = 0.9,
        stabilization_minutes: int = 5,
        monitoring_interval: float = 2.0,
        trend_window_minutes: int = 10,
    ):
        """Initialize proactive scaler.

        Args:
            resource_monitor: ResourceMonitor instance for metrics
            tier_detector: HardwareTierDetector for tier-based thresholds
            upgrade_threshold: Resource usage threshold for upgrades (default 0.8 = 80%)
            downgrade_threshold: Resource usage threshold for downgrades (default 0.9 = 90%)
            stabilization_minutes: Minimum time between upgrades (default 5 minutes)
            monitoring_interval: Background monitoring interval in seconds
            trend_window_minutes: Window for trend analysis in minutes
        """
        self.logger = logging.getLogger(__name__)

        # Core dependencies
        self.resource_monitor = resource_monitor or ResourceMonitor()
        self.tier_detector = tier_detector or HardwareTierDetector()

        # Configuration
        self.upgrade_threshold = upgrade_threshold
        self.downgrade_threshold = downgrade_threshold
        self.stabilization_seconds = stabilization_minutes * 60
        self.monitoring_interval = monitoring_interval
        self.trend_window_seconds = trend_window_minutes * 60

        # State management
        self._monitoring_active = False
        self._monitoring_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()

        # Resource history and trend analysis
        self._resource_history: deque = deque(maxlen=500)  # Store last 500 samples
        self._performance_metrics: deque = deque(maxlen=100)  # Last 100 operations
        self._scaling_history: List[ScalingEvent] = []

        # Stabilization tracking
        self._last_upgrade_time: float = 0
        self._last_downgrade_time: float = 0
        self._current_model_size: Optional[str] = None
        self._stabilization_cooldown: bool = False

        # Callbacks for external systems
        self._on_scaling_decision: Optional[Callable[[ScalingEvent], None]] = None

        # Hysteresis to prevent thrashing
        self._hysteresis_margin = 0.05  # 5% margin between upgrade/downgrade

        self.logger.info("ProactiveScaler initialized with hybrid monitoring")

    def set_scaling_callback(self, callback: Callable[[ScalingEvent], None]) -> None:
        """Set callback function for scaling decisions.

        Args:
            callback: Function to call when scaling decision is made
        """
        self._on_scaling_decision = callback

    def start_continuous_monitoring(self) -> None:
        """Start background continuous monitoring."""
        if self._monitoring_active:
            self.logger.warning("Monitoring already active")
            return

        self._monitoring_active = True
        self._shutdown_event.clear()

        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True, name="ProactiveScaler-Monitor"
        )
        self._monitoring_thread.start()

        self.logger.info("Started continuous background monitoring")

    def stop_continuous_monitoring(self) -> None:
        """Stop background continuous monitoring."""
        if not self._monitoring_active:
            return

        self._monitoring_active = False
        self._shutdown_event.set()

        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5.0)

        self.logger.info("Stopped continuous background monitoring")

    def check_preflight_resources(
        self, operation_type: str = "model_inference"
    ) -> Tuple[bool, str]:
        """Perform quick pre-flight resource check before operation.

        Args:
            operation_type: Type of operation being attempted

        Returns:
            Tuple of (can_proceed, reason_if_denied)
        """
        try:
            resources = self.resource_monitor.get_current_resources()

            # Critical resource checks
            if resources["memory_percent"] > self.downgrade_threshold * 100:
                return (
                    False,
                    f"Memory usage too high: {resources['memory_percent']:.1f}%",
                )

            if resources["cpu_percent"] > self.downgrade_threshold * 100:
                return False, f"CPU usage too high: {resources['cpu_percent']:.1f}%"

            # Check for immediate degradation needs
            if self._should_immediate_degrade(resources):
                return (
                    False,
                    "Immediate degradation required - resources critically constrained",
                )

            return True, "Resources adequate for operation"

        except Exception as e:
            self.logger.error(f"Error in pre-flight check: {e}")
            return False, f"Pre-flight check failed: {e}"

    def should_upgrade_model(
        self, current_resources: Optional[Dict[str, float]] = None
    ) -> bool:
        """Check if conditions allow for model upgrade.

        Args:
            current_resources: Current resource snapshot (optional)

        Returns:
            True if upgrade conditions are met
        """
        try:
            resources = (
                current_resources or self.resource_monitor.get_current_resources()
            )
            current_time = time.time()

            # Check stabilization cooldown
            if current_time - self._last_upgrade_time < self.stabilization_seconds:
                return False

            # Check if resources are consistently low enough for upgrade
            if not self._resources_support_upgrade(resources):
                return False

            # Analyze trends to ensure stability
            if not self._trend_supports_upgrade():
                return False

            # Check if we're in stabilization cooldown from previous downgrades
            if self._stabilization_cooldown:
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error checking upgrade conditions: {e}")
            return False

    def initiate_graceful_degradation(
        self, reason: str, immediate: bool = False
    ) -> Optional[str]:
        """Initiate graceful degradation to smaller model.

        Args:
            reason: Reason for degradation
            immediate: Whether degradation should happen immediately

        Returns:
            Recommended smaller model size or None
        """
        try:
            resources = self.resource_monitor.get_current_resources()
            current_tier = self.tier_detector.detect_current_tier()
            tier_config = self.tier_detector.get_tier_config(current_tier)

            # Determine target model size based on current constraints
            if self._current_model_size == "large":
                target_size = "medium"
            elif self._current_model_size == "medium":
                target_size = "small"
            else:
                target_size = "small"  # Stay at small if already small

            # Check if degradation is beneficial
            if target_size == self._current_model_size:
                self.logger.warning(
                    "Already at minimum model size, cannot degrade further"
                )
                return None

            current_time = time.time()
            if not immediate:
                # Apply stabilization period for downgrades too
                if (
                    current_time - self._last_downgrade_time
                    < self.stabilization_seconds
                ):
                    self.logger.info("Degradation blocked by stabilization period")
                    return None

            # Create scaling event
            event = ScalingEvent(
                timestamp=current_time,
                decision=ScalingDecision.DOWNGRADE,
                old_model_size=self._current_model_size,
                new_model_size=target_size,
                reason=reason,
                resources=resources,
                tier=current_tier,
            )

            # Record the decision
            self._record_scaling_decision(event)

            # Update timing
            self._last_downgrade_time = current_time
            self._current_model_size = target_size

            self.logger.info(
                f"Initiated graceful degradation to {target_size}: {reason}"
            )

            # Trigger callback if set
            if self._on_scaling_decision:
                self._on_scaling_decision(event)

            return target_size

        except Exception as e:
            self.logger.error(f"Error initiating degradation: {e}")
            return None

    def analyze_resource_trends(self) -> Dict[str, Any]:
        """Analyze resource usage trends for predictive scaling.

        Returns:
            Dictionary with trend analysis and predictions
        """
        try:
            if len(self._resource_history) < 10:
                return {"status": "insufficient_data"}

            # Calculate trends for key metrics
            memory_trend = self._calculate_trend(
                [entry["memory"] for entry in self._resource_history]
            )
            cpu_trend = self._calculate_trend(
                [entry["cpu"] for entry in self._resource_history]
            )

            # Predict future usage based on trends
            future_memory = self._predict_future_usage(memory_trend)
            future_cpu = self._predict_future_usage(cpu_trend)

            # Determine scaling recommendation
            recommendation = self._generate_trend_recommendation(
                memory_trend, cpu_trend, future_memory, future_cpu
            )

            return {
                "status": "analyzed",
                "memory_trend": memory_trend,
                "cpu_trend": cpu_trend,
                "predicted_memory_usage": future_memory,
                "predicted_cpu_usage": future_cpu,
                "recommendation": recommendation,
                "confidence": self._calculate_trend_confidence(),
            }

        except Exception as e:
            self.logger.error(f"Error analyzing trends: {e}")
            return {"status": "error", "error": str(e)}

    def update_performance_metrics(
        self, operation_type: str, duration_ms: float, success: bool
    ) -> None:
        """Update performance metrics for scaling decisions.

        Args:
            operation_type: Type of operation performed
            duration_ms: Duration in milliseconds
            success: Whether operation was successful
        """
        metric = {
            "timestamp": time.time(),
            "operation_type": operation_type,
            "duration_ms": duration_ms,
            "success": success,
        }

        self._performance_metrics.append(metric)

        # Keep only recent metrics (maintained by deque maxlen)

    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status and recommendations.

        Returns:
            Dictionary with scaling status information
        """
        try:
            current_resources = self.resource_monitor.get_current_resources()
            current_tier = self.tier_detector.detect_current_tier()

            return {
                "monitoring_active": self._monitoring_active,
                "current_model_size": self._current_model_size,
                "current_tier": current_tier,
                "current_resources": current_resources,
                "upgrade_available": self.should_upgrade_model(current_resources),
                "degradation_needed": self._should_immediate_degrade(current_resources),
                "stabilization_cooldown": self._stabilization_cooldown,
                "last_upgrade_time": self._last_upgrade_time,
                "last_downgrade_time": self._last_downgrade_time,
                "recent_decisions": self._scaling_history[-5:],  # Last 5 decisions
                "trend_analysis": self.analyze_resource_trends(),
            }

        except Exception as e:
            self.logger.error(f"Error getting scaling status: {e}")
            return {"status": "error", "error": str(e)}

    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        self.logger.info("Starting proactive scaling monitoring loop")

        while not self._shutdown_event.wait(self.monitoring_interval):
            try:
                if not self._monitoring_active:
                    break

                # Collect current resources
                resources = self.resource_monitor.get_current_resources()
                timestamp = time.time()

                # Update resource history
                self._update_resource_history(resources, timestamp)

                # Check for scaling opportunities
                self._check_scaling_opportunities(resources, timestamp)

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1.0)  # Brief pause on error

        self.logger.info("Proactive scaling monitoring loop stopped")

    def _update_resource_history(
        self, resources: Dict[str, float], timestamp: float
    ) -> None:
        """Update resource history with current snapshot."""
        history_entry = {
            "timestamp": timestamp,
            "memory": resources["memory_percent"],
            "cpu": resources["cpu_percent"],
            "available_memory_gb": resources["available_memory_gb"],
            "gpu_utilization": resources.get("gpu_utilization_percent", 0),
        }

        self._resource_history.append(history_entry)

        # Also update the resource monitor's history
        self.resource_monitor.update_history()

    def _check_scaling_opportunities(
        self, resources: Dict[str, float], timestamp: float
    ) -> None:
        """Check for proactive scaling opportunities."""
        try:
            # Check for immediate degradation needs
            if self._should_immediate_degrade(resources):
                degradation_reason = f"Critical resource usage: Memory {resources['memory_percent']:.1f}%, CPU {resources['cpu_percent']:.1f}%"
                self.initiate_graceful_degradation(degradation_reason, immediate=True)
                return

            # Check for upgrade opportunities
            if self.should_upgrade_model(resources):
                if not self._stabilization_cooldown:
                    upgrade_recommendation = self._determine_upgrade_target()
                    if upgrade_recommendation:
                        self._execute_upgrade(
                            upgrade_recommendation, resources, timestamp
                        )

            # Update stabilization cooldown status
            self._update_stabilization_status()

        except Exception as e:
            self.logger.error(f"Error checking scaling opportunities: {e}")

    def _should_immediate_degrade(self, resources: Dict[str, float]) -> bool:
        """Check if immediate degradation is required."""
        # Critical thresholds that require immediate action
        memory_critical = resources["memory_percent"] > self.downgrade_threshold * 100
        cpu_critical = resources["cpu_percent"] > self.downgrade_threshold * 100

        # Also check available memory (avoid OOM)
        memory_low = resources["available_memory_gb"] < 1.0  # Less than 1GB available

        return memory_critical or cpu_critical or memory_low

    def _resources_support_upgrade(self, resources: Dict[str, float]) -> bool:
        """Check if current resources support model upgrade."""
        memory_ok = resources["memory_percent"] < self.upgrade_threshold * 100
        cpu_ok = resources["cpu_percent"] < self.upgrade_threshold * 100
        memory_available = (
            resources["available_memory_gb"] >= 4.0
        )  # Need at least 4GB free

        return memory_ok and cpu_ok and memory_available

    def _trend_supports_upgrade(self) -> bool:
        """Check if resource trends support model upgrade."""
        if len(self._resource_history) < 20:  # Need more data
            return False

        # Analyze recent trends
        recent_entries = list(self._resource_history)[-20:]

        memory_values = [entry["memory"] for entry in recent_entries]
        cpu_values = [entry["cpu"] for entry in recent_entries]

        memory_trend = self._calculate_trend(memory_values)
        cpu_trend = self._calculate_trend(cpu_values)

        # Only upgrade if trends are stable or decreasing
        return memory_trend in ["stable", "decreasing"] and cpu_trend in [
            "stable",
            "decreasing",
        ]

    def _determine_upgrade_target(self) -> Optional[str]:
        """Determine the best upgrade target based on current tier."""
        try:
            current_tier = self.tier_detector.detect_current_tier()
            preferred_models = self.tier_detector.get_preferred_models(current_tier)

            if not preferred_models:
                return None

            # Find next larger model in preferred list
            size_order = ["small", "medium", "large"]
            current_idx = (
                size_order.index(self._current_model_size)
                if self._current_model_size
                else -1
            )

            # Find the largest model we can upgrade to
            for size in reversed(size_order):  # Check large to small
                if size in preferred_models and size_order.index(size) > current_idx:
                    return size

            return None

        except Exception as e:
            self.logger.error(f"Error determining upgrade target: {e}")
            return None

    def _execute_upgrade(
        self, target_size: str, resources: Dict[str, float], timestamp: float
    ) -> None:
        """Execute model upgrade with proper recording."""
        try:
            current_time = time.time()

            # Check stabilization period
            if current_time - self._last_upgrade_time < self.stabilization_seconds:
                self.logger.debug("Upgrade blocked by stabilization period")
                return

            # Create scaling event
            event = ScalingEvent(
                timestamp=current_time,
                decision=ScalingDecision.UPGRADE,
                old_model_size=self._current_model_size,
                new_model_size=target_size,
                reason=f"Proactive upgrade based on resource availability: {resources['memory_percent']:.1f}% memory, {resources['cpu_percent']:.1f}% CPU",
                resources=resources,
                tier=self.tier_detector.detect_current_tier(),
            )

            # Record the decision
            self._record_scaling_decision(event)

            # Update state
            self._last_upgrade_time = current_time
            self._current_model_size = target_size

            # Set stabilization cooldown
            self._stabilization_cooldown = True

            self.logger.info(f"Executed proactive upgrade to {target_size}")

            # Trigger callback if set
            if self._on_scaling_decision:
                self._on_scaling_decision(event)

        except Exception as e:
            self.logger.error(f"Error executing upgrade: {e}")

    def _update_stabilization_status(self) -> None:
        """Update stabilization cooldown status."""
        current_time = time.time()

        # Check if stabilization period has passed
        time_since_last_change = min(
            current_time - self._last_upgrade_time,
            current_time - self._last_downgrade_time,
        )

        if time_since_last_change > self.stabilization_seconds:
            self._stabilization_cooldown = False
        else:
            self._stabilization_cooldown = True

    def _record_scaling_decision(self, event: ScalingEvent) -> None:
        """Record a scaling decision in history."""
        self._scaling_history.append(event)

        # Keep only recent history (last 50 decisions)
        if len(self._scaling_history) > 50:
            self._scaling_history = self._scaling_history[-50:]

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from a list of values."""
        if len(values) < 5:
            return "insufficient_data"

        # Simple linear regression for trend
        n = len(values)
        x_values = list(range(n))

        sum_x = sum(x_values)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(x_values, values))
        sum_x2 = sum(x * x for x in x_values)

        # Calculate slope
        try:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)

            # Determine trend based on slope magnitude
            if abs(slope) < 0.1:
                return "stable"
            elif slope > 0:
                return "increasing"
            else:
                return "decreasing"
        except ZeroDivisionError:
            return "stable"

    def _predict_future_usage(self, trend: str) -> Optional[float]:
        """Predict future resource usage based on trend."""
        if trend == "stable":
            return None  # No change predicted
        elif trend == "increasing":
            # Predict usage in 5 minutes based on current trend
            return min(0.95, 0.8 + 0.1)  # Conservative estimate
        elif trend == "decreasing":
            return max(0.3, 0.6 - 0.1)  # Conservative estimate

        return None

    def _generate_trend_recommendation(
        self,
        memory_trend: str,
        cpu_trend: str,
        future_memory: Optional[float],
        future_cpu: Optional[float],
    ) -> str:
        """Generate scaling recommendation based on trend analysis."""
        if memory_trend == "increasing" or cpu_trend == "increasing":
            return "monitor_closely"  # Resources trending up
        elif memory_trend == "decreasing" and cpu_trend == "decreasing":
            return "consider_upgrade"  # Resources trending down
        elif memory_trend == "stable" and cpu_trend == "stable":
            return "maintain_current"  # Stable conditions
        else:
            return "monitor_closely"  # Mixed signals

    def _calculate_trend_confidence(self) -> float:
        """Calculate confidence in trend predictions."""
        if len(self._resource_history) < 20:
            return 0.3  # Low confidence with limited data

        # Higher confidence with more data and stable trends
        data_factor = min(1.0, len(self._resource_history) / 100.0)

        # Calculate consistency of recent trends
        recent_entries = list(self._resource_history)[-20:]
        memory_variance = self._calculate_variance(
            [entry["memory"] for entry in recent_entries]
        )
        cpu_variance = self._calculate_variance(
            [entry["cpu"] for entry in recent_entries]
        )

        # Lower variance = higher confidence
        variance_factor = max(0.3, 1.0 - (memory_variance + cpu_variance) / 200.0)

        return data_factor * variance_factor

    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values."""
        if not values:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance

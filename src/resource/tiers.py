"""Hardware tier detection and management system."""

import os
import yaml
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from ..models.resource_monitor import ResourceMonitor


class HardwareTierDetector:
    """Detects and classifies hardware capabilities into performance tiers.

    This class loads configurable tier definitions and uses system resource
    monitoring to classify the current system into appropriate tiers for
    intelligent model selection.
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize hardware tier detector.

        Args:
            config_path: Path to tier configuration file. If None, uses default.
        """
        self.logger = logging.getLogger(__name__)

        # Set default config path relative to this file
        if config_path is None:
            config_path = (
                Path(__file__).parent.parent / "config" / "resource_tiers.yaml"
            )

        self.config_path = Path(config_path)
        self.tier_config: Optional[Dict[str, Any]] = None
        self.resource_monitor = ResourceMonitor()

        # Cache tier detection result
        self._cached_tier: Optional[str] = None
        self._cache_time: float = 0
        self._cache_duration: float = 60.0  # Cache for 1 minute

        # Load configuration
        self._load_tier_config()

    def _load_tier_config(self) -> None:
        """Load tier definitions from YAML configuration file.

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid
        """
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.tier_config = yaml.safe_load(f)
            self.logger.info(f"Loaded tier configuration from {self.config_path}")
        except FileNotFoundError:
            self.logger.error(f"Tier configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            self.logger.error(f"Invalid YAML in tier configuration: {e}")
            raise

    def detect_current_tier(self) -> str:
        """Determine system tier based on current resources.

        Returns:
            Tier name: 'low_end', 'mid_range', or 'high_end'
        """
        # Check cache first
        import time

        current_time = time.time()
        if (
            self._cached_tier is not None
            and current_time - self._cache_time < self._cache_duration
        ):
            return self._cached_tier

        try:
            resources = self.resource_monitor.get_current_resources()
            tier = self._classify_resources(resources)

            # Cache result
            self._cached_tier = tier
            self._cache_time = current_time

            self.logger.info(f"Detected hardware tier: {tier}")
            return tier

        except Exception as e:
            self.logger.error(f"Failed to detect tier: {e}")
            return "low_end"  # Conservative fallback

    def _classify_resources(self, resources: Dict[str, float]) -> str:
        """Classify system resources into tier based on configuration.

        Args:
            resources: Current system resources from ResourceMonitor

        Returns:
            Tier classification
        """
        if not self.tier_config or "tiers" not in self.tier_config:
            self.logger.error("No tier configuration loaded")
            return "low_end"

        tiers = self.tier_config["tiers"]

        # Extract key metrics
        ram_gb = resources.get("available_memory_gb", 0)
        cpu_cores = os.cpu_count() or 1
        gpu_vram_gb = resources.get("gpu_free_vram_gb", 0)
        gpu_total_vram_gb = resources.get("gpu_total_vram_gb", 0)

        self.logger.debug(
            f"Resources: RAM={ram_gb:.1f}GB, CPU={cpu_cores}, GPU={gpu_total_vram_gb:.1f}GB"
        )

        # Check tiers in order: high_end -> mid_range -> low_end
        for tier_name in ["high_end", "mid_range", "low_end"]:
            if tier_name not in tiers:
                continue

            tier_config = tiers[tier_name]

            if self._meets_tier_requirements(
                tier_config, ram_gb, cpu_cores, gpu_vram_gb, gpu_total_vram_gb
            ):
                return tier_name

        return "low_end"  # Conservative fallback

    def _meets_tier_requirements(
        self,
        tier_config: Dict[str, Any],
        ram_gb: float,
        cpu_cores: int,
        gpu_vram_gb: float,
        gpu_total_vram_gb: float,
    ) -> bool:
        """Check if system meets tier requirements.

        Args:
            tier_config: Configuration for the tier to check
            ram_gb: Available system RAM in GB
            cpu_cores: Number of CPU cores
            gpu_vram_gb: Available GPU VRAM in GB
            gpu_total_vram_gb: Total GPU VRAM in GB

        Returns:
            True if system meets all requirements for this tier
        """
        try:
            # Check RAM requirements
            ram_req = tier_config.get("ram_gb", {})
            ram_min = ram_req.get("min", 0)
            ram_max = ram_req.get("max")

            if ram_gb < ram_min:
                return False
            if ram_max is not None and ram_gb > ram_max:
                return False

            # Check CPU core requirements
            cpu_req = tier_config.get("cpu_cores", {})
            cpu_min = cpu_req.get("min", 1)
            cpu_max = cpu_req.get("max")

            if cpu_cores < cpu_min:
                return False
            if cpu_max is not None and cpu_cores > cpu_max:
                return False

            # Check GPU requirements
            gpu_required = tier_config.get("gpu_required", False)
            if gpu_required:
                gpu_vram_req = tier_config.get("gpu_vram_gb", {}).get("min", 0)
                if gpu_total_vram_gb < gpu_vram_req:
                    return False
            elif gpu_total_vram_gb > 0:  # GPU present but not required
                gpu_vram_max = tier_config.get("gpu_vram_gb", {}).get("max")
                if gpu_vram_max is not None and gpu_total_vram_gb > gpu_vram_max:
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error checking tier requirements: {e}")
            return False

    def get_tier_config(self, tier_name: Optional[str] = None) -> Dict[str, Any]:
        """Get configuration for a specific tier.

        Args:
            tier_name: Tier to get config for. If None, uses detected tier.

        Returns:
            Tier configuration dictionary
        """
        if tier_name is None:
            tier_name = self.detect_current_tier()

        if not self.tier_config or "tiers" not in self.tier_config:
            return {}

        return self.tier_config["tiers"].get(tier_name, {})

    def get_preferred_models(self, tier_name: Optional[str] = None) -> List[str]:
        """Get preferred model list for detected or specified tier.

        Args:
            tier_name: Tier to get models for. If None, uses detected tier.

        Returns:
            List of preferred model sizes for the tier
        """
        tier_config = self.get_tier_config(tier_name)
        return tier_config.get("preferred_models", ["small"])

    def get_scaling_thresholds(
        self, tier_name: Optional[str] = None
    ) -> Dict[str, float]:
        """Get scaling thresholds for detected or specified tier.

        Args:
            tier_name: Tier to get thresholds for. If None, uses detected tier.

        Returns:
            Dictionary with memory_percent and cpu_percent thresholds
        """
        tier_config = self.get_tier_config(tier_name)
        return tier_config.get(
            "scaling_thresholds", {"memory_percent": 75.0, "cpu_percent": 80.0}
        )

    def is_gpu_required(self, tier_name: Optional[str] = None) -> bool:
        """Check if detected or specified tier requires GPU.

        Args:
            tier_name: Tier to check. If None, uses detected tier.

        Returns:
            True if GPU is required for this tier
        """
        tier_config = self.get_tier_config(tier_name)
        return tier_config.get("gpu_required", False)

    def get_performance_characteristics(
        self, tier_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get performance characteristics for detected or specified tier.

        Args:
            tier_name: Tier to get characteristics for. If None, uses detected tier.

        Returns:
            Dictionary with performance characteristics
        """
        tier_config = self.get_tier_config(tier_name)
        return tier_config.get("performance_characteristics", {})

    def can_upgrade_model(
        self, current_model_size: str, target_model_size: str
    ) -> bool:
        """Check if system can handle a larger model.

        Args:
            current_model_size: Current model size (e.g., 'small', 'medium')
            target_model_size: Target model size (e.g., 'medium', 'large')

        Returns:
            True if system can handle the target model size
        """
        preferred_models = self.get_preferred_models()

        # If target model is in preferred list, system should handle it
        if target_model_size in preferred_models:
            return True

        # Check if target is larger than current but still within capabilities
        size_order = ["small", "medium", "large"]
        try:
            current_idx = size_order.index(current_model_size)
            target_idx = size_order.index(target_model_size)

            # Only allow upgrade if target is in preferred models
            return target_idx <= max(
                [
                    size_order.index(size)
                    for size in preferred_models
                    if size in size_order
                ]
            )

        except ValueError:
            return False

    def get_model_recommendations(self) -> Dict[str, Any]:
        """Get comprehensive model recommendations for current system.

        Returns:
            Dictionary with model recommendations and capabilities
        """
        tier = self.detect_current_tier()
        tier_config = self.get_tier_config(tier)

        return {
            "detected_tier": tier,
            "preferred_models": self.get_preferred_models(tier),
            "model_size_range": tier_config.get("model_size_range", {}),
            "performance_characteristics": self.get_performance_characteristics(tier),
            "scaling_thresholds": self.get_scaling_thresholds(tier),
            "gpu_required": self.is_gpu_required(tier),
            "description": tier_config.get("description", ""),
        }

    def refresh_config(self) -> None:
        """Reload tier configuration from file.

        Useful for runtime configuration updates without restarting.
        """
        self._load_tier_config()
        self._cached_tier = None  # Clear cache to force re-detection

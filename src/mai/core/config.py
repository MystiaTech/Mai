"""
Configuration management system for Mai.

Handles loading, validation, and management of all Mai settings
with proper defaults and runtime updates.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
import copy
import threading

# Import exceptions
try:
    from .exceptions import ConfigFileError, ConfigValidationError, ConfigMissingError
except ImportError:
    # Define placeholder exceptions if module not available
    class ConfigFileError(Exception):
        pass

    class ConfigValidationError(Exception):
        pass

    class ConfigMissingError(Exception):
        pass


@dataclass
class ModelConfig:
    """Model-specific configuration."""

    preferred_models: list = field(
        default_factory=lambda: ["llama2", "mistral", "codellama", "vicuna"]
    )
    fallback_models: list = field(default_factory=lambda: ["llama2:7b", "mistral:7b", "phi"])
    resource_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "cpu_warning": 0.8,
            "cpu_critical": 0.95,
            "ram_warning": 0.8,
            "ram_critical": 0.95,
            "gpu_warning": 0.9,
            "gpu_critical": 0.98,
        }
    )
    context_windows: Dict[str, int] = field(
        default_factory=lambda: {
            "llama2": 4096,
            "mistral": 8192,
            "codellama": 16384,
            "vicuna": 4096,
            "phi": 2048,
        }
    )
    auto_switch: bool = True
    switch_threshold: float = 0.7  # Performance degradation threshold


@dataclass
class ResourceConfig:
    """Resource monitoring configuration."""

    monitoring_enabled: bool = True
    check_interval: float = 5.0  # seconds
    trend_window: int = 60  # seconds for trend analysis
    performance_history_size: int = 100
    gpu_detection: bool = True
    fallback_detection: bool = True
    resource_warnings: bool = True
    conservative_estimates: bool = True
    memory_buffer: float = 0.5  # 50% buffer for context overhead


@dataclass
class ContextConfig:
    """Context management configuration."""

    compression_enabled: bool = True
    warning_threshold: float = 0.75  # Warn at 75% of context
    critical_threshold: float = 0.90  # Critical at 90%
    budget_ratio: float = 0.9  # Budget at 90% of context
    max_conversation_length: int = 100
    preserve_key_elements: bool = True
    compression_cache_ttl: int = 3600  # 1 hour
    min_quality_score: float = 0.7


@dataclass
class GitConfig:
    """Git workflow configuration."""

    auto_commit: bool = True
    commit_grouping: bool = True
    natural_language_messages: bool = True
    staging_branch: str = "mai-staging"
    auto_merge: bool = True
    health_checks: bool = True
    stability_test_duration: int = 300  # 5 minutes
    auto_revert: bool = True
    commit_delay: float = 10.0  # seconds between commits


@dataclass
class LoggingConfig:
    """Logging and debugging configuration."""

    level: str = "INFO"
    file_logging: bool = True
    console_logging: bool = True
    log_file: str = "logs/mai.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    debug_mode: bool = False
    performance_logging: bool = True
    error_tracking: bool = True


@dataclass
class MemoryConfig:
    """Memory system and compression configuration."""

    # Compression thresholds
    message_count: int = 50
    age_days: int = 30
    memory_limit_mb: int = 500

    # Summarization settings
    summarization_model: str = "llama2"
    preserve_elements: list = field(
        default_factory=lambda: ["preferences", "decisions", "patterns", "key_facts"]
    )
    min_quality_score: float = 0.7
    max_summary_length: int = 1000
    context_messages: int = 30

    # Adaptive weighting
    importance_decay_days: int = 90
    pattern_weight: float = 1.5
    technical_weight: float = 1.2
    planning_weight: float = 1.3
    recency_boost: float = 1.2
    keyword_boost: float = 1.5

    # Strategy settings
    keep_recent_count: int = 10
    max_patterns_extracted: int = 5
    topic_extraction_method: str = "keyword"
    pattern_confidence_threshold: float = 0.6

    # Retrieval settings
    similarity_threshold: float = 0.7
    max_results: int = 5
    include_content: bool = False
    semantic_weight: float = 0.4
    keyword_weight: float = 0.3
    recency_weight: float = 0.2
    user_pattern_weight: float = 0.1

    # Performance settings
    max_memory_usage_mb: int = 200
    max_cpu_usage_percent: int = 80
    max_compression_time_seconds: int = 30
    enable_background_compression: bool = True
    compression_interval_hours: int = 6
    batch_size: int = 5


@dataclass
class Config:
    """Main configuration class for Mai."""

    models: ModelConfig = field(default_factory=ModelConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    git: GitConfig = field(default_factory=GitConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    # Runtime state
    config_file: Optional[str] = None
    last_modified: Optional[float] = None
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def __post_init__(self):
        """Initialize configuration after dataclass creation."""
        # Ensure log directory exists
        if self.logging.file_logging:
            log_path = Path(self.logging.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)


class ConfigManager:
    """
    Configuration manager with loading, validation, and hot-reload capabilities.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to configuration file (YAML or JSON)
        """
        self.config_path = config_path
        self.config = Config()
        self._observers = []
        self._lock = threading.RLock()

        # Load configuration if path provided
        if config_path:
            self.load_config(config_path)

        # Apply environment variable overrides
        self._apply_env_overrides()

    def load_config(self, config_path: Optional[str] = None) -> Config:
        """
        Load configuration from file.

        Args:
            config_path: Path to configuration file

        Returns:
            Loaded Config object

        Raises:
            ConfigFileError: If file cannot be loaded
            ConfigValidationError: If configuration is invalid
        """
        if config_path:
            self.config_path = config_path

        if not self.config_path or not os.path.exists(self.config_path):
            # Use default configuration
            self.config = Config()
            return self.config

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                if self.config_path.endswith(".yaml") or self.config_path.endswith(".yml"):
                    data = yaml.safe_load(f)
                elif self.config_path.endswith(".json"):
                    data = json.load(f)
                else:
                    raise ConfigFileError(f"Unsupported config format: {self.config_path}")

            # Merge with defaults
            self.config = self._merge_with_defaults(data)
            self.config.config_file = self.config_path
            self.config.last_modified = os.path.getmtime(self.config_path)

            # Validate configuration
            self._validate_config()

            # Apply environment overrides
            self._apply_env_overrides()

            # Notify observers
            self._notify_observers("config_loaded", self.config)

            return self.config

        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise ConfigFileError(f"Invalid configuration file format: {e}")
        except Exception as e:
            raise ConfigFileError(f"Error loading configuration: {e}")

    def save_config(self, config_path: Optional[str] = None) -> bool:
        """
        Save current configuration to file.

        Args:
            config_path: Path to save configuration (uses current if None)

        Returns:
            True if saved successfully

        Raises:
            ConfigFileError: If file cannot be saved
        """
        if config_path:
            self.config_path = config_path

        if not self.config_path:
            raise ConfigFileError("No configuration path specified")

        try:
            # Ensure directory exists
            config_dir = os.path.dirname(self.config_path)
            if config_dir:
                os.makedirs(config_dir, exist_ok=True)

            # Convert to dictionary
            config_dict = asdict(self.config)

            # Remove runtime state
            config_dict.pop("config_file", None)
            config_dict.pop("last_modified", None)
            config_dict.pop("_lock", None)

            # Save with comments (YAML format preferred)
            with open(self.config_path, "w", encoding="utf-8") as f:
                if self.config_path.endswith(".yaml") or self.config_path.endswith(".yml"):
                    # Add comments for documentation
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config_dict, f, indent=2)

            self.config.last_modified = os.path.getmtime(self.config_path)

            # Notify observers
            self._notify_observers("config_saved", self.config)

            return True

        except Exception as e:
            raise ConfigFileError(f"Error saving configuration: {e}")

    def get_model_config(self) -> ModelConfig:
        """Get model-specific configuration."""
        return self.config.models

    def get_resource_config(self) -> ResourceConfig:
        """Get resource monitoring configuration."""
        return self.config.resources

    def get_context_config(self) -> ContextConfig:
        """Get context management configuration."""
        return self.config.context

    def get_git_config(self) -> GitConfig:
        """Get git workflow configuration."""
        return self.config.git

    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration."""
        return self.config.logging

    def get_memory_config(self) -> MemoryConfig:
        """Get memory configuration."""
        return self.config.memory

    def update_config(self, updates: Dict[str, Any], section: Optional[str] = None) -> bool:
        """
        Update configuration with new values.

        Args:
            updates: Dictionary of updates to apply
            section: Configuration section to update (optional)

        Returns:
            True if updated successfully

        Raises:
            ConfigValidationError: If updates are invalid
        """
        with self._lock:
            # Store old values for rollback
            old_values = {}

            try:
                # Apply updates
                if section:
                    if hasattr(self.config, section):
                        section_config = getattr(self.config, section)
                        for key, value in updates.items():
                            if hasattr(section_config, key):
                                old_values[f"{section}.{key}"] = getattr(section_config, key)
                                setattr(section_config, key, value)
                            else:
                                raise ConfigValidationError(f"Invalid config key: {section}.{key}")
                    else:
                        raise ConfigValidationError(f"Invalid config section: {section}")
                else:
                    # Apply to root config
                    for key, value in updates.items():
                        if hasattr(self.config, key):
                            old_values[key] = getattr(self.config, key)
                            setattr(self.config, key, value)
                        else:
                            raise ConfigValidationError(f"Invalid config key: {key}")

                # Validate updated configuration
                self._validate_config()

                # Save if file path available
                if self.config_path:
                    self.save_config()

                # Notify observers
                self._notify_observers("config_updated", self.config, old_values)

                return True

            except Exception as e:
                # Rollback changes on error
                for path, value in old_values.items():
                    if "." in path:
                        section, key = path.split(".", 1)
                        if hasattr(self.config, section):
                            setattr(getattr(self.config, section), key, value)
                    else:
                        setattr(self.config, path, value)
                raise ConfigValidationError(f"Invalid configuration update: {e}")

    def reload_config(self) -> bool:
        """
        Reload configuration from file.

        Returns:
            True if reloaded successfully
        """
        if not self.config_path:
            return False

        try:
            return self.load_config(self.config_path) is not None
        except Exception:
            return False

    def add_observer(self, callback):
        """
        Add observer for configuration changes.

        Args:
            callback: Function to call on config changes
        """
        with self._lock:
            self._observers.append(callback)

    def remove_observer(self, callback):
        """
        Remove observer for configuration changes.

        Args:
            callback: Function to remove
        """
        with self._lock:
            if callback in self._observers:
                self._observers.remove(callback)

    def _merge_with_defaults(self, data: Dict[str, Any]) -> Config:
        """
        Merge loaded data with default configuration.

        Args:
            data: Loaded configuration data

        Returns:
            Merged Config object
        """
        # Start with defaults
        default_dict = asdict(Config())

        # Recursively merge
        merged = self._deep_merge(default_dict, data)

        # Create Config from merged dict
        return Config(**merged)

    def _deep_merge(self, default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries.

        Args:
            default: Default values
            override: Override values

        Returns:
            Merged dictionary
        """
        result = copy.deepcopy(default)

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def _validate_config(self):
        """
        Validate configuration values.

        Raises:
            ConfigValidationError: If configuration is invalid
        """
        # Validate model config
        if not self.config.models.preferred_models:
            raise ConfigValidationError("No preferred models configured")

        if not 0 <= self.config.models.switch_threshold <= 1:
            raise ConfigValidationError("Model switch threshold must be between 0 and 1")

        # Validate resource config
        if not 0 < self.config.resources.check_interval <= 60:
            raise ConfigValidationError("Resource check interval must be between 0 and 60 seconds")

        # Validate context config
        if not 0 < self.config.context.budget_ratio <= 1:
            raise ConfigValidationError("Context budget ratio must be between 0 and 1")

        if (
            not 0
            < self.config.context.warning_threshold
            < self.config.context.critical_threshold
            <= 1
        ):
            raise ConfigValidationError("Invalid context thresholds: warning < critical <= 1")

        # Validate logging config
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.config.logging.level not in valid_levels:
            raise ConfigValidationError(f"Invalid log level: {self.config.logging.level}")

    def _apply_env_overrides(self):
        """Apply environment variable overrides."""
        # Model overrides
        if "MAI_PREFERRED_MODELS" in os.environ:
            models = [m.strip() for m in os.environ["MAI_PREFERRED_MODELS"].split(",")]
            self.config.models.preferred_models = models

        if "MAI_AUTO_SWITCH" in os.environ:
            self.config.models.auto_switch = os.environ["MAI_AUTO_SWITCH"].lower() == "true"

        # Resource overrides
        if "MAI_RESOURCE_MONITORING" in os.environ:
            self.config.resources.monitoring_enabled = (
                os.environ["MAI_RESOURCE_MONITORING"].lower() == "true"
            )

        # Context overrides
        if "MAI_CONTEXT_BUDGET_RATIO" in os.environ:
            try:
                ratio = float(os.environ["MAI_CONTEXT_BUDGET_RATIO"])
                if 0 < ratio <= 1:
                    self.config.context.budget_ratio = ratio
            except ValueError:
                pass

        # Logging overrides
        if "MAI_DEBUG_MODE" in os.environ:
            self.config.logging.debug_mode = os.environ["MAI_DEBUG_MODE"].lower() == "true"

        # Memory overrides
        if "MAI_MEMORY_LIMIT_MB" in os.environ:
            try:
                limit = int(os.environ["MAI_MEMORY_LIMIT_MB"])
                if limit > 0:
                    self.config.memory.memory_limit_mb = limit
            except ValueError:
                pass

        if "MAI_COMPRESSION_MODEL" in os.environ:
            self.config.memory.summarization_model = os.environ["MAI_COMPRESSION_MODEL"]

        if "MAI_ENABLE_BACKGROUND_COMPRESSION" in os.environ:
            self.config.memory.enable_background_compression = (
                os.environ["MAI_ENABLE_BACKGROUND_COMPRESSION"].lower() == "true"
            )

    def _notify_observers(self, event: str, *args):
        """Notify observers of configuration changes."""
        for observer in self._observers:
            try:
                observer(event, *args)
            except Exception:
                # Don't let observer errors break config management
                pass

    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get summary of current configuration.

        Returns:
            Dictionary with configuration summary
        """
        return {
            "config_file": self.config.config_file,
            "last_modified": self.config.last_modified,
            "models": {
                "preferred_count": len(self.config.models.preferred_models),
                "auto_switch": self.config.models.auto_switch,
                "switch_threshold": self.config.models.switch_threshold,
            },
            "resources": {
                "monitoring_enabled": self.config.resources.monitoring_enabled,
                "check_interval": self.config.resources.check_interval,
                "gpu_detection": self.config.resources.gpu_detection,
            },
            "context": {
                "compression_enabled": self.config.context.compression_enabled,
                "budget_ratio": self.config.context.budget_ratio,
                "warning_threshold": self.config.context.warning_threshold,
            },
            "git": {
                "auto_commit": self.config.git.auto_commit,
                "staging_branch": self.config.git.staging_branch,
                "auto_merge": self.config.git.auto_merge,
            },
            "logging": {
                "level": self.config.logging.level,
                "file_logging": self.config.logging.file_logging,
                "debug_mode": self.config.logging.debug_mode,
            },
            "memory": {
                "message_count": self.config.memory.message_count,
                "age_days": self.config.memory.age_days,
                "memory_limit_mb": self.config.memory.memory_limit_mb,
                "summarization_model": self.config.memory.summarization_model,
                "enable_background_compression": self.config.memory.enable_background_compression,
            },
        }


# Global configuration manager instance
_config_manager = None


def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """
    Get global configuration manager instance.

    Args:
        config_path: Path to configuration file (only used on first call)

    Returns:
        ConfigManager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    return _config_manager


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get current configuration.

    Args:
        config_path: Optional path to configuration file (only used on first call)

    Returns:
        Current Config object
    """
    return get_config_manager(config_path).config


def load_memory_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load memory-specific configuration from YAML file.

    Args:
        config_path: Path to memory configuration file

    Returns:
        Dictionary with memory configuration settings
    """
    # Default memory config path
    if config_path is None:
        config_path = os.path.join(".mai", "config", "memory.yaml")

    # If file doesn't exist, return default settings
    if not os.path.exists(config_path):
        return {
            "compression": {
                "thresholds": {"message_count": 50, "age_days": 30, "memory_limit_mb": 500}
            }
        }

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            if config_path.endswith((".yaml", ".yml")):
                config_data = yaml.safe_load(f)
            else:
                config_data = json.load(f)

        # Validate and merge with defaults
        default_config = {
            "compression": {
                "thresholds": {"message_count": 50, "age_days": 30, "memory_limit_mb": 500},
                "summarization": {
                    "model": "llama2",
                    "preserve_elements": ["preferences", "decisions", "patterns", "key_facts"],
                    "min_quality_score": 0.7,
                    "max_summary_length": 1000,
                    "context_messages": 30,
                },
            }
        }

        # Deep merge with defaults
        merged_config = _deep_merge(default_config, config_data)

        # Validate key memory settings
        compression_config = merged_config.get("compression", {})
        thresholds = compression_config.get("thresholds", {})

        if thresholds.get("message_count", 0) < 10:
            raise ConfigValidationError(
                field_name="message_count",
                field_value=thresholds.get("message_count"),
                validation_error="must be at least 10",
            )

        if thresholds.get("age_days", 0) < 1:
            raise ConfigValidationError(
                field_name="age_days",
                field_value=thresholds.get("age_days"),
                validation_error="must be at least 1 day",
            )

        if thresholds.get("memory_limit_mb", 0) < 100:
            raise ConfigValidationError(
                field_name="memory_limit_mb",
                field_value=thresholds.get("memory_limit_mb"),
                validation_error="must be at least 100MB",
            )

        return merged_config

    except (yaml.YAMLError, json.JSONDecodeError) as e:
        raise ConfigFileError(
            file_path=config_path,
            operation="load_memory_config",
            error_details=f"Invalid format: {e}",
        )
    except Exception as e:
        raise ConfigFileError(
            file_path=config_path,
            operation="load_memory_config",
            error_details=f"Error loading: {e}",
        )

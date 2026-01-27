"""
Custom exception hierarchy for Mai error handling.

Provides clear, actionable error information for all Mai components
with context data and resolution suggestions.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import traceback
import time


@dataclass
class ErrorContext:
    """Context information for errors."""

    component: str  # Component where error occurred
    operation: str  # Operation being performed
    data: Dict[str, Any]  # Relevant context data
    timestamp: float = field(default_factory=time.time)  # When error occurred
    user_friendly: bool = True  # Whether to show to users


class MaiError(Exception):
    """
    Base exception for all Mai-specific errors.

    All Mai exceptions should inherit from this class to provide
    consistent error handling and context.
    """

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        suggestions: Optional[List[str]] = None,
        cause: Optional[Exception] = None,
    ):
        """
        Initialize Mai error.

        Args:
            message: Error message
            error_code: Unique error code for programmatic handling
            context: Error context information
            suggestions: Suggestions for resolution
            cause: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or ErrorContext(
            component="unknown", operation="unknown", data={}, timestamp=time.time()
        )
        self.suggestions = suggestions or []
        self.cause = cause
        self.severity = self._determine_severity()

    def _determine_severity(self) -> str:
        """Determine error severity based on type and context."""
        if (
            "Critical" in self.__class__.__name__
            or self.error_code
            and "CRITICAL" in self.error_code
        ):
            return "critical"
        elif (
            "Warning" in self.__class__.__name__ or self.error_code and "WARNING" in self.error_code
        ):
            return "warning"
        else:
            return "error"

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "severity": self.severity,
            "context": {
                "component": self.context.component,
                "operation": self.context.operation,
                "data": self.context.data,
                "timestamp": self.context.timestamp,
                "user_friendly": self.context.user_friendly,
            },
            "suggestions": self.suggestions,
            "cause": str(self.cause) if self.cause else None,
            "traceback": traceback.format_exc() if self.severity == "critical" else None,
        }

    def __str__(self) -> str:
        """String representation of error."""
        return self.message


class ModelError(MaiError):
    """Base class for model-related errors."""

    def __init__(self, message: str, model_name: Optional[str] = None, **kwargs):
        kwargs.setdefault(
            "context",
            ErrorContext(
                component="model_interface",
                operation="model_operation",
                data={"model_name": model_name} if model_name else {},
            ),
        )
        super().__init__(message, **kwargs)
        self.model_name = model_name


class ModelNotFoundError(ModelError):
    """Raised when requested model is not available."""

    def __init__(self, model_name: str, available_models: Optional[List[str]] = None):
        suggestions = [
            f"Check if '{model_name}' is installed in Ollama",
            "Run 'ollama list' to see available models",
            "Try downloading the model with 'ollama pull'",
        ]
        if available_models:
            suggestions.append(f"Available models: {', '.join(available_models[:5])}")

        super().__init__(
            f"Model '{model_name}' not found",
            model_name=model_name,
            error_code="MODEL_NOT_FOUND",
            suggestions=suggestions,
        )
        self.available_models = available_models or []


class ModelSwitchError(ModelError):
    """Raised when model switching fails."""

    def __init__(self, from_model: str, to_model: str, reason: Optional[str] = None):
        message = f"Failed to switch from '{from_model}' to '{to_model}'"
        if reason:
            message += f": {reason}"

        suggestions = [
            "Check if target model is available",
            "Verify sufficient system resources for target model",
            "Try switching to a smaller model first",
        ]

        super().__init__(
            message,
            model_name=to_model,
            error_code="MODEL_SWITCH_FAILED",
            context=ErrorContext(
                component="model_switcher",
                operation="switch_model",
                data={"from_model": from_model, "to_model": to_model, "reason": reason},
            ),
            suggestions=suggestions,
        )
        self.from_model = from_model
        self.to_model = to_model


class ModelConnectionError(ModelError):
    """Raised when cannot connect to Ollama or model service."""

    def __init__(self, service_url: str, timeout: Optional[float] = None):
        message = f"Cannot connect to model service at {service_url}"
        if timeout:
            message += f" (timeout: {timeout}s)"

        suggestions = [
            "Check if Ollama is running",
            f"Verify service URL: {service_url}",
            "Check network connectivity",
            "Try restarting Ollama service",
        ]

        super().__init__(
            message,
            error_code="MODEL_CONNECTION_FAILED",
            context=ErrorContext(
                component="ollama_client",
                operation="connect",
                data={"service_url": service_url, "timeout": timeout},
            ),
            suggestions=suggestions,
        )
        self.service_url = service_url
        self.timeout = timeout


class ModelInferenceError(ModelError):
    """Raised when model inference request fails."""

    def __init__(self, model_name: str, prompt_length: int, error_details: Optional[str] = None):
        message = f"Inference failed for model '{model_name}'"
        if error_details:
            message += f": {error_details}"

        suggestions = [
            "Check if model is loaded properly",
            "Try with a shorter prompt",
            "Verify model context window limits",
            "Check available system memory",
        ]

        super().__init__(
            message,
            model_name=model_name,
            error_code="MODEL_INFERENCE_FAILED",
            context=ErrorContext(
                component="model_interface",
                operation="inference",
                data={
                    "model_name": model_name,
                    "prompt_length": prompt_length,
                    "error_details": error_details,
                },
            ),
            suggestions=suggestions,
        )
        self.prompt_length = prompt_length
        self.error_details = error_details


class ResourceError(MaiError):
    """Base class for resource-related errors."""

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault(
            "context",
            ErrorContext(component="resource_monitor", operation="resource_check", data={}),
        )
        super().__init__(message, **kwargs)


class ResourceExhaustedError(ResourceError):
    """Raised when system resources are depleted."""

    def __init__(self, resource_type: str, current_usage: float, limit: float):
        message = (
            f"Resource '{resource_type}' exhausted: {current_usage:.1%} used (limit: {limit:.1%})"
        )

        suggestions = [
            "Close other applications to free up resources",
            "Try using a smaller model",
            "Wait for resources to become available",
            "Consider upgrading system resources",
        ]

        super().__init__(
            message,
            error_code="RESOURCE_EXHAUSTED",
            context=ErrorContext(
                component="resource_monitor",
                operation="check_resources",
                data={
                    "resource_type": resource_type,
                    "current_usage": current_usage,
                    "limit": limit,
                    "excess": current_usage - limit,
                },
            ),
            suggestions=suggestions,
        )
        self.resource_type = resource_type
        self.current_usage = current_usage
        self.limit = limit


class ResourceMonitorError(ResourceError):
    """Raised when resource monitoring fails."""

    def __init__(self, operation: str, error_details: Optional[str] = None):
        message = f"Resource monitoring failed during {operation}"
        if error_details:
            message += f": {error_details}"

        suggestions = [
            "Check if monitoring dependencies are installed",
            "Verify system permissions for resource access",
            "Try using fallback monitoring methods",
            "Restart the application",
        ]

        super().__init__(
            message,
            error_code="RESOURCE_MONITOR_FAILED",
            context=ErrorContext(
                component="resource_monitor",
                operation=operation,
                data={"error_details": error_details},
            ),
            suggestions=suggestions,
        )
        self.operation = operation
        self.error_details = error_details


class InsufficientMemoryError(ResourceError):
    """Raised when insufficient memory for operation."""

    def __init__(self, required_memory: int, available_memory: int, operation: str):
        message = f"Insufficient memory for {operation}: need {required_memory}MB, have {available_memory}MB"

        suggestions = [
            "Close other applications to free memory",
            "Try with a smaller model or context",
            "Increase swap space if available",
            "Consider using a model with lower memory requirements",
        ]

        super().__init__(
            message,
            error_code="INSUFFICIENT_MEMORY",
            context=ErrorContext(
                component="memory_manager",
                operation="allocate_memory",
                data={
                    "required_memory": required_memory,
                    "available_memory": available_memory,
                    "shortfall": required_memory - available_memory,
                    "operation": operation,
                },
            ),
            suggestions=suggestions,
        )
        self.required_memory = required_memory
        self.available_memory = available_memory
        self.operation = operation


class ContextError(MaiError):
    """Base class for context-related errors."""

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault(
            "context",
            ErrorContext(component="context_manager", operation="context_operation", data={}),
        )
        super().__init__(message, **kwargs)


class ContextTooLongError(ContextError):
    """Raised when conversation exceeds context window limits."""

    def __init__(self, current_tokens: int, max_tokens: int, model_name: str):
        message = (
            f"Conversation too long for {model_name}: {current_tokens} tokens (max: {max_tokens})"
        )

        suggestions = [
            "Enable context compression",
            "Remove older messages from conversation",
            "Use a model with larger context window",
            "Split conversation into smaller parts",
        ]

        super().__init__(
            message,
            error_code="CONTEXT_TOO_LONG",
            context=ErrorContext(
                component="context_compressor",
                operation="validate_context",
                data={
                    "current_tokens": current_tokens,
                    "max_tokens": max_tokens,
                    "excess": current_tokens - max_tokens,
                    "model_name": model_name,
                },
            ),
            suggestions=suggestions,
        )
        self.current_tokens = current_tokens
        self.max_tokens = max_tokens
        self.model_name = model_name


class ContextCompressionError(ContextError):
    """Raised when context compression fails."""

    def __init__(
        self, original_tokens: int, target_ratio: float, error_details: Optional[str] = None
    ):
        message = (
            f"Context compression failed: {original_tokens} tokens → target {target_ratio:.1%}"
        )
        if error_details:
            message += f": {error_details}"

        suggestions = [
            "Try with a higher compression ratio",
            "Check if conversation contains valid text",
            "Verify compression quality thresholds",
            "Use manual message removal instead",
        ]

        super().__init__(
            message,
            error_code="CONTEXT_COMPRESSION_FAILED",
            context=ErrorContext(
                component="context_compressor",
                operation="compress",
                data={
                    "original_tokens": original_tokens,
                    "target_ratio": target_ratio,
                    "error_details": error_details,
                },
            ),
            suggestions=suggestions,
        )
        self.original_tokens = original_tokens
        self.target_ratio = target_ratio
        self.error_details = error_details


class ContextCorruptionError(ContextError):
    """Raised when context data is invalid or corrupted."""

    def __init__(self, context_type: str, corruption_details: Optional[str] = None):
        message = f"Context corruption detected in {context_type}"
        if corruption_details:
            message += f": {corruption_details}"

        suggestions = [
            "Clear conversation history and start fresh",
            "Verify context serialization format",
            "Check for data encoding issues",
            "Rebuild context from valid messages",
        ]

        super().__init__(
            message,
            error_code="CONTEXT_CORRUPTION",
            context=ErrorContext(
                component="context_manager",
                operation="validate_context",
                data={"context_type": context_type, "corruption_details": corruption_details},
            ),
            suggestions=suggestions,
        )
        self.context_type = context_type
        self.corruption_details = corruption_details


class GitError(MaiError):
    """Base class for Git-related errors."""

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault(
            "context", ErrorContext(component="git_interface", operation="git_operation", data={})
        )
        super().__init__(message, **kwargs)


class GitRepositoryError(GitError):
    """Raised for Git repository issues."""

    def __init__(self, repo_path: str, error_details: Optional[str] = None):
        message = f"Git repository error in {repo_path}"
        if error_details:
            message += f": {error_details}"

        suggestions = [
            "Verify directory is a Git repository",
            "Check Git repository permissions",
            "Run 'git status' to diagnose issues",
            "Initialize repository with 'git init' if needed",
        ]

        super().__init__(
            message,
            error_code="GIT_REPOSITORY_ERROR",
            context=ErrorContext(
                component="git_interface",
                operation="validate_repository",
                data={"repo_path": repo_path, "error_details": error_details},
            ),
            suggestions=suggestions,
        )
        self.repo_path = repo_path
        self.error_details = error_details


class GitCommitError(GitError):
    """Raised when commit operation fails."""

    def __init__(
        self, operation: str, files: Optional[List[str]] = None, error_details: Optional[str] = None
    ):
        message = f"Git {operation} failed"
        if error_details:
            message += f": {error_details}"

        suggestions = [
            "Check if files exist and are readable",
            "Verify write permissions for repository",
            "Run 'git status' to check repository state",
            "Stage files with 'git add' before committing",
        ]

        super().__init__(
            message,
            error_code="GIT_COMMIT_FAILED",
            context=ErrorContext(
                component="git_committer",
                operation=operation,
                data={"files": files or [], "error_details": error_details},
            ),
            suggestions=suggestions,
        )
        self.operation = operation
        self.files = files or []
        self.error_details = error_details


class GitMergeError(GitError):
    """Raised for merge conflicts or failures."""

    def __init__(
        self,
        branch_name: str,
        conflict_files: Optional[List[str]] = None,
        error_details: Optional[str] = None,
    ):
        message = f"Git merge failed for branch '{branch_name}'"
        if error_details:
            message += f": {error_details}"

        suggestions = [
            "Resolve merge conflicts manually",
            "Use 'git status' to see conflicted files",
            "Consider using 'git merge --abort' to cancel",
            "Pull latest changes before merging",
        ]

        super().__init__(
            message,
            error_code="GIT_MERGE_FAILED",
            context=ErrorContext(
                component="git_workflow",
                operation="merge",
                data={
                    "branch_name": branch_name,
                    "conflict_files": conflict_files or [],
                    "error_details": error_details,
                },
            ),
            suggestions=suggestions,
        )
        self.branch_name = branch_name
        self.conflict_files = conflict_files or []
        self.error_details = error_details


class ConfigurationError(MaiError):
    """Base class for configuration-related errors."""

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault(
            "context",
            ErrorContext(component="config_manager", operation="config_operation", data={}),
        )
        super().__init__(message, **kwargs)


class ConfigFileError(ConfigurationError):
    """Raised for configuration file issues."""

    def __init__(self, file_path: str, operation: str, error_details: Optional[str] = None):
        message = f"Configuration file error during {operation}: {file_path}"
        if error_details:
            message += f": {error_details}"

        suggestions = [
            "Verify file path and permissions",
            "Check file format (YAML/JSON)",
            "Ensure file contains valid configuration",
            "Create default configuration file if missing",
        ]

        super().__init__(
            message,
            error_code="CONFIG_FILE_ERROR",
            context=ErrorContext(
                component="config_manager",
                operation=operation,
                data={"file_path": file_path, "error_details": error_details},
            ),
            suggestions=suggestions,
        )
        self.file_path = file_path
        self.operation = operation
        self.error_details = error_details


class ConfigValidationError(ConfigurationError):
    """Raised for invalid configuration values."""

    def __init__(self, field_name: str, field_value: Any, validation_error: str):
        message = (
            f"Invalid configuration value for '{field_name}': {field_value} - {validation_error}"
        )

        suggestions = [
            "Check configuration documentation for valid values",
            "Verify value type and range constraints",
            "Use default configuration values",
            "Check for typos in field names",
        ]

        super().__init__(
            message,
            error_code="CONFIG_VALIDATION_FAILED",
            context=ErrorContext(
                component="config_manager",
                operation="validate_config",
                data={
                    "field_name": field_name,
                    "field_value": str(field_value),
                    "validation_error": validation_error,
                },
            ),
            suggestions=suggestions,
        )
        self.field_name = field_name
        self.field_value = field_value
        self.validation_error = validation_error


class ConfigMissingError(ConfigurationError):
    """Raised when required configuration is missing."""

    def __init__(self, missing_keys: List[str], config_section: Optional[str] = None):
        section_msg = f" in section '{config_section}'" if config_section else ""
        message = f"Required configuration missing{section_msg}: {', '.join(missing_keys)}"

        suggestions = [
            "Add missing keys to configuration file",
            "Check configuration documentation for required fields",
            "Use default configuration as template",
            "Verify configuration file is being loaded correctly",
        ]

        super().__init__(
            message,
            error_code="CONFIG_MISSING_REQUIRED",
            context=ErrorContext(
                component="config_manager",
                operation="check_requirements",
                data={"missing_keys": missing_keys, "config_section": config_section},
            ),
            suggestions=suggestions,
        )
        self.missing_keys = missing_keys
        self.config_section = config_section


# Error handling utilities


def format_error_for_user(error: MaiError) -> str:
    """
    Convert technical error to user-friendly message.

    Args:
        error: MaiError instance

    Returns:
        User-friendly error message
    """
    if not isinstance(error, MaiError):
        return f"Unexpected error: {str(error)}"

    # Use the message if it's user-friendly
    if error.context.user_friendly:
        return str(error)

    # Create user-friendly version
    friendly_message = error.message

    # Remove technical details
    technical_terms = ["traceback", "exception", "error_code", "context"]
    for term in technical_terms:
        friendly_message = friendly_message.lower().replace(term, "")

    # Add top suggestion
    if error.suggestions:
        friendly_message += f"\n\nSuggestion: {error.suggestions[0]}"

    return friendly_message.strip()


def is_retriable_error(error: Exception) -> bool:
    """
    Determine if error can be retried.

    Args:
        error: Exception instance

    Returns:
        True if error is retriable
    """
    if isinstance(error, MaiError):
        retriable_codes = [
            "MODEL_CONNECTION_FAILED",
            "RESOURCE_MONITOR_FAILED",
            "CONTEXT_COMPRESSION_FAILED",
        ]
        return error.error_code in retriable_codes

    # Non-Mai errors: only retry network/connection issues
    error_str = str(error).lower()
    retriable_patterns = ["connection", "timeout", "network", "temporary", "unavailable"]

    return any(pattern in error_str for pattern in retriable_patterns)


def get_error_severity(error: Exception) -> str:
    """
    Classify error severity.

    Args:
        error: Exception instance

    Returns:
        Severity level: 'warning', 'error', or 'critical'
    """
    if isinstance(error, MaiError):
        return error.severity

    # Classify non-Mai errors
    error_str = str(error).lower()

    if any(pattern in error_str for pattern in ["critical", "fatal"]):
        return "critical"
    elif any(pattern in error_str for pattern in ["warning"]):
        return "warning"
    else:
        return "error"


def create_error_context(component: str, operation: str, **data) -> ErrorContext:
    """
    Create error context with current timestamp.

    Args:
        component: Component name
        operation: Operation name
        **data: Additional context data

    Returns:
        ErrorContext instance
    """
    return ErrorContext(component=component, operation=operation, data=data, timestamp=time.time())


# Exception handler for logging and monitoring


class ErrorHandler:
    """
    Central error handler for Mai components.

    Provides consistent error logging, metrics, and user notification.
    """

    def __init__(self, logger=None):
        """
        Initialize error handler.

        Args:
            logger: Logger instance for error reporting
        """
        self.logger = logger
        self.error_counts = {}
        self.last_errors = {}

    def handle_error(self, error: Exception, component: str = "unknown"):
        """
        Handle error with logging and metrics.

        Args:
            error: Exception to handle
            component: Component where error occurred
        """
        # Count errors
        error_type = error.__class__.__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        self.last_errors[error_type] = {
            "error": error,
            "component": component,
            "timestamp": time.time(),
        }

        # Log error
        if self.logger:
            severity = get_error_severity(error)
            if severity == "critical":
                self.logger.critical(f"Critical error in {component}: {error}")
            elif severity == "error":
                self.logger.error(f"Error in {component}: {error}")
            else:
                self.logger.warning(f"Warning in {component}: {error}")

        # Return formatted error for user
        if isinstance(error, MaiError):
            return format_error_for_user(error)
        else:
            return f"An error occurred in {component}: {str(error)}"

    def get_error_stats(self) -> Dict[str, Any]:
        """
        Get error statistics.

        Returns:
            Dictionary with error statistics
        """
        return {
            "error_counts": self.error_counts.copy(),
            "last_errors": {
                k: {
                    "error": str(v["error"]),
                    "component": v["component"],
                    "timestamp": v["timestamp"],
                }
                for k, v in self.last_errors.items()
            },
            "total_errors": sum(self.error_counts.values()),
        }

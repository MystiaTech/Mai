"""
Git workflow management for Mai's self-improvement system.

Provides staging branch management, validation, and cleanup
capabilities for safe code improvements.
"""

from .workflow import StagingWorkflow
from .committer import AutoCommitter
from .health_check import HealthChecker

__all__ = ["StagingWorkflow", "AutoCommitter", "HealthChecker"]

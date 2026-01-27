"""
Staging workflow management for Mai's self-improvement system.

Handles branch creation, management, and cleanup for testing improvements
before merging to main codebase.
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path

try:
    from git import Repo, InvalidGitRepositoryError, GitCommandError, Head
    from git.exc import GitError
except ImportError:
    raise ImportError("GitPython is required. Install with: pip install GitPython")

from ..core import MaiError, ConfigurationError


class StagingWorkflowError(MaiError):
    """Raised when staging workflow operations fail."""

    pass


class StagingWorkflow:
    """
    Manages staging branches for safe code improvements.

    Provides branch creation, validation, and cleanup capabilities
    with proper error handling and recovery.
    """

    def __init__(self, project_path: str = ".", timeout: int = 30):
        """
        Initialize staging workflow.

        Args:
            project_path: Path to git repository
            timeout: Timeout for git operations in seconds

        Raises:
            ConfigurationError: If not a git repository
        """
        self.project_path = Path(project_path).resolve()
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)

        try:
            self.repo = Repo(self.project_path)
        except InvalidGitRepositoryError:
            raise ConfigurationError(f"Not a git repository: {self.project_path}")

        # Configure retry logic for git operations
        self.max_retries = 3
        self.retry_delay = 1

        # Branch naming pattern
        self.branch_prefix = "staging"

        # Initialize health check integration (will be connected later)
        self.health_checker = None

        self.logger.info(f"Staging workflow initialized for {self.project_path}")

    def create_staging_branch(self, improvement_type: str, description: str) -> Dict[str, Any]:
        """
        Create a staging branch for improvements.

        Args:
            improvement_type: Type of improvement (e.g., 'optimization', 'feature', 'bugfix')
            description: Description of improvement

        Returns:
            Dictionary with branch information

        Raises:
            StagingWorkflowError: If branch creation fails
        """
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        # Sanitize description for branch name
        description_safe = "".join(c for c in description[:20] if c.isalnum() or c in "-_").lower()
        branch_name = f"{self.branch_prefix}/{improvement_type}-{timestamp}-{description_safe}"

        try:
            # Ensure we're on main/develop branch
            self._ensure_main_branch()

            # Check if branch already exists
            if branch_name in [ref.name for ref in self.repo.refs]:
                self.logger.warning(f"Branch {branch_name} already exists")
                existing_branch = self.repo.refs[branch_name]
                return {
                    "branch_name": branch_name,
                    "branch": existing_branch,
                    "created": False,
                    "message": f"Branch {branch_name} already exists",
                }

            # Create new branch
            current_branch = self.repo.active_branch
            new_branch = self.repo.create_head(branch_name, current_branch.commit.hexsha)

            # Simple metadata handling - just log for now
            self.logger.info(f"Branch metadata: type={improvement_type}, desc={description}")

            self.logger.info(f"Created staging branch: {branch_name}")

            return {
                "branch_name": branch_name,
                "branch": new_branch,
                "created": True,
                "timestamp": timestamp,
                "improvement_type": improvement_type,
                "description": description,
                "message": f"Created staging branch {branch_name}",
            }

        except (GitError, GitCommandError) as e:
            raise StagingWorkflowError(f"Failed to create branch {branch_name}: {e}")

    def switch_to_branch(self, branch_name: str) -> Dict[str, Any]:
        """
        Safely switch to specified branch.

        Args:
            branch_name: Name of branch to switch to

        Returns:
            Dictionary with switch result

        Raises:
            StagingWorkflowError: If switch fails
        """
        try:
            # Check for uncommitted changes
            if self.repo.is_dirty(untracked_files=True):
                return {
                    "success": False,
                    "branch_name": branch_name,
                    "message": "Working directory has uncommitted changes. Commit or stash first.",
                    "uncommitted": True,
                }

            # Verify branch exists
            if branch_name not in [ref.name for ref in self.repo.refs]:
                return {
                    "success": False,
                    "branch_name": branch_name,
                    "message": f"Branch {branch_name} does not exist",
                    "exists": False,
                }

            # Switch to branch
            branch = self.repo.refs[branch_name]
            branch.checkout()

            self.logger.info(f"Switched to branch: {branch_name}")

            return {
                "success": True,
                "branch_name": branch_name,
                "message": f"Switched to {branch_name}",
                "current_commit": str(self.repo.active_branch.commit),
            }

        except (GitError, GitCommandError) as e:
            raise StagingWorkflowError(f"Failed to switch to branch {branch_name}: {e}")

    def get_active_staging_branches(self) -> List[Dict[str, Any]]:
        """
        List all staging branches with metadata.

        Returns:
            List of dictionaries with branch information
        """
        staging_branches = []
        current_time = datetime.now()

        for ref in self.repo.refs:
            if ref.name.startswith(self.branch_prefix + "/"):
                try:
                    # Get branch age
                    commit_time = datetime.fromtimestamp(ref.commit.committed_date)
                    age = current_time - commit_time

                    # Check if branch is stale (> 7 days)
                    is_stale = age > timedelta(days=7)

                    # Simple metadata for now
                    metadata = {
                        "improvement_type": "unknown",
                        "description": "no description",
                        "created": "unknown",
                    }

                    staging_branches.append(
                        {
                            "name": ref.name,
                            "commit": str(ref.commit),
                            "commit_message": ref.commit.message.strip(),
                            "created": commit_time.isoformat(),
                            "age_days": age.days,
                            "age_hours": age.total_seconds() / 3600,
                            "is_stale": is_stale,
                            "metadata": metadata,
                            "is_current": ref.name == self.repo.active_branch.name,
                        }
                    )

                except Exception as e:
                    self.logger.warning(f"Error processing branch {ref.name}: {e}")
                    continue

        # Sort by creation time (newest first)
        staging_branches.sort(key=lambda x: x["created"], reverse=True)
        return staging_branches

    def validate_branch_state(self, branch_name: str) -> Dict[str, Any]:
        """
        Validate branch state for safe merging.

        Args:
            branch_name: Name of branch to validate

        Returns:
            Dictionary with validation results
        """
        try:
            if branch_name not in [ref.name for ref in self.repo.refs]:
                return {
                    "valid": False,
                    "branch_name": branch_name,
                    "issues": [f"Branch {branch_name} does not exist"],
                    "can_merge": False,
                }

            # Switch to branch temporarily if not already there
            original_branch = self.repo.active_branch.name
            if original_branch != branch_name:
                switch_result = self.switch_to_branch(branch_name)
                if not switch_result["success"]:
                    return {
                        "valid": False,
                        "branch_name": branch_name,
                        "issues": [switch_result["message"]],
                        "can_merge": False,
                    }

            issues = []

            # Check for uncommitted changes
            if self.repo.is_dirty(untracked_files=True):
                issues.append("Working directory has uncommitted changes")

            # Check for merge conflicts with main branch
            try:
                # Try to simulate merge without actually merging
                main_branch = self._get_main_branch()
                if main_branch and branch_name != main_branch:
                    merge_base = self.repo.merge_base(branch_name, main_branch)
                    if not merge_base:
                        issues.append("No common ancestor with main branch")
            except Exception as e:
                issues.append(f"Cannot determine merge compatibility: {e}")

            # Switch back to original branch
            if original_branch != branch_name:
                self.switch_to_branch(original_branch)

            return {
                "valid": len(issues) == 0,
                "branch_name": branch_name,
                "issues": issues,
                "can_merge": len(issues) == 0,
                "metadata": {"improvement_type": "unknown", "description": "no description"},
            }

        except Exception as e:
            return {
                "valid": False,
                "branch_name": branch_name,
                "issues": [f"Validation failed: {e}"],
                "can_merge": False,
            }

    def cleanup_staging_branch(
        self, branch_name: str, keep_if_failed: bool = False
    ) -> Dict[str, Any]:
        """
        Clean up staging branch after merge or when abandoned.

        Args:
            branch_name: Name of branch to cleanup
            keep_if_failed: Keep branch if validation failed

        Returns:
            Dictionary with cleanup result
        """
        try:
            if branch_name not in [ref.name for ref in self.repo.refs]:
                return {
                    "success": False,
                    "branch_name": branch_name,
                    "message": f"Branch {branch_name} does not exist",
                }

            # Check validation result if keep_if_failed is True
            if keep_if_failed:
                validation = self.validate_branch_state(branch_name)
                if not validation["can_merge"]:
                    return {
                        "success": False,
                        "branch_name": branch_name,
                        "message": "Keeping branch due to validation failures",
                        "validation": validation,
                    }

            # Don't delete current branch
            if branch_name == self.repo.active_branch.name:
                return {
                    "success": False,
                    "branch_name": branch_name,
                    "message": "Cannot delete currently active branch",
                }

            # Delete branch
            self.repo.delete_head(branch_name, force=True)

            self.logger.info(f"Cleaned up staging branch: {branch_name}")

            return {
                "success": True,
                "branch_name": branch_name,
                "message": f"Deleted staging branch {branch_name}",
                "deleted": True,
            }

        except (GitError, GitCommandError) as e:
            self.logger.error(f"Failed to cleanup branch {branch_name}: {e}")
            return {
                "success": False,
                "branch_name": branch_name,
                "message": f"Failed to delete branch: {e}",
                "deleted": False,
            }

    def cleanup_old_staging_branches(self, days_old: int = 7) -> Dict[str, Any]:
        """
        Clean up old staging branches.

        Args:
            days_old: Age threshold in days

        Returns:
            Dictionary with cleanup results
        """
        staging_branches = self.get_active_staging_branches()
        old_branches = [b for b in staging_branches if b["age_days"] > days_old]

        cleanup_results = []
        for branch_info in old_branches:
            result = self.cleanup_staging_branch(branch_info["name"])
            cleanup_results.append(result)

        successful = sum(1 for r in cleanup_results if r["success"])

        return {
            "total_old_branches": len(old_branches),
            "cleaned_up": successful,
            "failed": len(old_branches) - successful,
            "results": cleanup_results,
        }

    def _ensure_main_branch(self) -> None:
        """Ensure we're on main or develop branch."""
        current = self.repo.active_branch.name
        main_branch = self._get_main_branch()

        if main_branch and current != main_branch:
            try:
                self.repo.refs[main_branch].checkout()
            except (GitError, GitCommandError) as e:
                self.logger.warning(f"Cannot switch to {main_branch}: {e}")

    def _get_main_branch(self) -> Optional[str]:
        """Get main/develop branch name."""
        for branch_name in ["main", "develop", "master"]:
            if branch_name in [ref.name for ref in self.repo.refs]:
                return branch_name
        return None

    def set_health_checker(self, health_checker) -> None:
        """Set health checker integration."""
        self.health_checker = health_checker

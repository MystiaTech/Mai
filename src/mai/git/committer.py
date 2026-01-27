"""
Automated commit generation and management for Mai's self-improvement system.

Handles staging changes, generating user-focused commit messages,
and managing commit history with proper validation.
"""

import os
import re
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any, Set
from pathlib import Path

try:
    from git import Repo, InvalidGitRepositoryError, GitCommandError, Diff, GitError
except ImportError:
    raise ImportError("GitPython is required. Install with: pip install GitPython")

from ..core import MaiError, ConfigurationError


class AutoCommitterError(MaiError):
    """Raised when automated commit operations fail."""

    pass


class AutoCommitter:
    """
    Automates commit generation and management for Mai's improvements.

    Provides staging, commit message generation, and history management
    with user-focused impact descriptions.
    """

    def __init__(self, project_path: str = "."):
        """
        Initialize auto committer.

        Args:
            project_path: Path to git repository

        Raises:
            ConfigurationError: If not a git repository
        """
        self.project_path = Path(project_path).resolve()
        self.logger = logging.getLogger(__name__)

        try:
            self.repo = Repo(self.project_path)
        except InvalidGitRepositoryError:
            raise ConfigurationError(f"Not a git repository: {self.project_path}")

        # Commit message templates and patterns
        self.templates = {
            "performance": "Faster {operation} for {scenario}",
            "bugfix": "Fixed {issue} - {impact on user}",
            "feature": "Added {capability} - now you can {user benefit}",
            "optimization": "Improved {system} - {performance gain}",
            "refactor": "Cleaned up {component} - {improvement}",
            "security": "Enhanced security for {area} - {protection}",
            "compatibility": "Made Mai work better with {environment} - {benefit}",
        }

        # File patterns to ignore
        self.ignore_patterns = {
            "*.pyc",
            "*.pyo",
            "*.pyd",
            "__pycache__",
            ".git",
            ".pytest_cache",
            ".coverage",
            "htmlcov",
            "*.log",
            ".env",
            "*.tmp",
            "*.temp",
            "*.bak",
            ".DS_Store",
            "*.swp",
            "*~",
        }

        # Group patterns by system
        self.group_patterns = {
            "model": ["src/mai/model/", "*.model.*"],
            "git": ["src/mai/git/", "*.git.*"],
            "core": ["src/mai/core/", "*.core.*"],
            "memory": ["src/mai/memory/", "*.memory.*"],
            "safety": ["src/mai/safety/", "*.safety.*"],
            "personality": ["src/mai/personality/", "*.personality.*"],
            "interface": ["src/mai/interface/", "*.interface.*"],
            "config": ["*.toml", "*.yaml", "*.yml", "*.conf", ".env*"],
        }

        # Initialize user information
        self._init_user_info()

        self.logger.info(f"Auto committer initialized for {self.project_path}")

    def stage_changes(
        self, file_patterns: Optional[List[str]] = None, group_by: str = "system"
    ) -> Dict[str, Any]:
        """
        Stage changed files for commit with optional grouping.

        Args:
            file_patterns: Specific file patterns to stage
            group_by: How to group changes ("system", "directory", "none")

        Returns:
            Dictionary with staging results and groups
        """
        try:
            # Get changed files
            changed_files = self._get_changed_files()

            # Filter by patterns if specified
            if file_patterns:
                changed_files = [
                    f for f in changed_files if self._matches_pattern(f, file_patterns)
                ]

            # Filter out ignored files
            staged_files = [f for f in changed_files if not self._should_ignore_file(f)]

            # Stage the files
            self.repo.index.add(staged_files)

            # Group changes
            groups = self._group_changes(staged_files, group_by) if group_by != "none" else {}

            self.logger.info(f"Staged {len(staged_files)} files in {len(groups)} groups")

            return {
                "staged_files": staged_files,
                "groups": groups,
                "total_files": len(staged_files),
                "message": f"Staged {len(staged_files)} files for commit",
            }

        except (GitError, GitCommandError) as e:
            raise AutoCommitterError(f"Failed to stage changes: {e}")

    def generate_commit_message(
        self, changes: List[str], impact_description: str, improvement_type: str = "feature"
    ) -> str:
        """
        Generate user-focused commit message.

        Args:
            changes: List of changed files
            impact_description: Description of impact on user
            improvement_type: Type of improvement

        Returns:
            User-focused commit message
        """
        # Try to use template
        if improvement_type in self.templates:
            template = self.templates[improvement_type]

            # Extract context from changes
            context = self._extract_context_from_files(changes)

            # Fill template
            try:
                message = template.format(**context, **{"user benefit": impact_description})
            except KeyError:
                # Fall back to impact description
                message = impact_description
        else:
            message = impact_description

        # Ensure user-focused language
        message = self._make_user_focused(message)

        # Add technical details as second line
        if len(changes) <= 5:
            tech_details = f"Files: {', '.join([Path(f).name for f in changes[:3]])}"
            if len(changes) > 3:
                tech_details += f" (+{len(changes) - 3} more)"
            message = f"{message}\n\n{tech_details}"

        # Limit length
        if len(message) > 100:
            message = message[:97] + "..."

        return message

    def commit_changes(
        self, message: str, files: Optional[List[str]] = None, validate_before: bool = True
    ) -> Dict[str, Any]:
        """
        Create commit with generated message and optional validation.

        Args:
            message: Commit message
            files: Specific files to commit (stages all if None)
            validate_before: Run validation before committing

        Returns:
            Dictionary with commit results
        """
        try:
            # Validate if requested
            if validate_before:
                validation = self._validate_commit(message, files)
                if not validation["valid"]:
                    return {
                        "success": False,
                        "message": "Commit validation failed",
                        "validation": validation,
                        "commit_hash": None,
                    }

            # Stage files if specified
            if files:
                self.repo.index.add(files)

            # Check if there are staged changes
            if not self.repo.is_dirty(untracked_files=True) and not self.repo.index.diff("HEAD"):
                return {"success": False, "message": "No changes to commit", "commit_hash": None}

            # Create commit with metadata
            commit = self.repo.index.commit(
                message=message, author_date=datetime.now(), committer_date=datetime.now()
            )

            commit_hash = commit.hexsha

            self.logger.info(f"Created commit: {commit_hash[:8]} - {message[:50]}")

            return {
                "success": True,
                "message": f"Committed {commit_hash[:8]}",
                "commit_hash": commit_hash,
                "short_hash": commit_hash[:8],
                "full_message": message,
            }

        except (GitError, GitCommandError) as e:
            raise AutoCommitterError(f"Failed to create commit: {e}")

    def get_commit_history(
        self, limit: int = 10, filter_by: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve commit history with metadata.

        Args:
            limit: Maximum number of commits to retrieve
            filter_by: Filter criteria (author, date range, patterns)

        Returns:
            List of commit information
        """
        try:
            commits = []

            for commit in self.repo.iter_commits(max_count=limit):
                # Apply filters
                if filter_by:
                    if "author" in filter_by and filter_by["author"] not in commit.author.name:
                        continue
                    if "since" in filter_by and commit.committed_date < filter_by["since"]:
                        continue
                    if "until" in filter_by and commit.committed_date > filter_by["until"]:
                        continue
                    if "pattern" in filter_by and not re.search(
                        filter_by["pattern"], commit.message
                    ):
                        continue

                commits.append(
                    {
                        "hash": commit.hexsha,
                        "short_hash": commit.hexsha[:8],
                        "message": commit.message.strip(),
                        "author": commit.author.name,
                        "date": datetime.fromtimestamp(commit.committed_date).isoformat(),
                        "files_changed": len(commit.stats.files),
                        "insertions": commit.stats.total["insertions"],
                        "deletions": commit.stats.total["deletions"],
                        "impact": self._extract_impact_from_message(commit.message),
                    }
                )

            return commits

        except (GitError, GitCommandError) as e:
            raise AutoCommitterError(f"Failed to get commit history: {e}")

    def revert_commit(self, commit_hash: str, create_branch: bool = True) -> Dict[str, Any]:
        """
        Safely revert specified commit.

        Args:
            commit_hash: Hash of commit to revert
            create_branch: Create backup branch before reverting

        Returns:
            Dictionary with revert results
        """
        try:
            # Validate commit exists
            try:
                commit = self.repo.commit(commit_hash)
            except Exception:
                return {
                    "success": False,
                    "message": f"Commit {commit_hash[:8]} not found",
                    "commit_hash": None,
                }

            # Create backup branch if requested
            backup_branch = None
            if create_branch:
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                backup_branch = f"backup/before-revert-{commit_hash[:8]}-{timestamp}"
                self.repo.create_head(backup_branch, self.repo.active_branch.commit)
                self.logger.info(f"Created backup branch: {backup_branch}")

            # Perform revert
            revert_commit = self.repo.git.revert("--no-edit", commit_hash)

            # Get new commit hash
            new_commit_hash = self.repo.head.commit.hexsha

            self.logger.info(f"Reverted commit {commit_hash[:8]} -> {new_commit_hash[:8]}")

            return {
                "success": True,
                "message": f"Reverted {commit_hash[:8]} successfully",
                "original_commit": commit_hash,
                "new_commit_hash": new_commit_hash,
                "new_short_hash": new_commit_hash[:8],
                "backup_branch": backup_branch,
                "original_message": commit.message.strip(),
            }

        except (GitError, GitCommandError) as e:
            raise AutoCommitterError(f"Failed to revert commit: {e}")

    def _get_changed_files(self) -> List[str]:
        """Get list of changed files in working directory."""
        changed_files = set()

        # Unstaged changes
        for item in self.repo.index.diff(None):
            changed_files.add(item.a_path)

        # Staged changes
        for item in self.repo.index.diff("HEAD"):
            changed_files.add(item.a_path)

        # Untracked files
        changed_files.update(self.repo.untracked_files)

        return list(changed_files)

    def _should_ignore_file(self, file_path: str) -> bool:
        """Check if file should be ignored."""
        file_name = Path(file_path).name

        for pattern in self.ignore_patterns:
            if self._matches_pattern(file_path, [pattern]):
                return True

        return False

    def _matches_pattern(self, file_path: str, patterns: List[str]) -> bool:
        """Check if file path matches any pattern."""
        import fnmatch

        for pattern in patterns:
            if fnmatch.fnmatch(file_path, pattern) or fnmatch.fnmatch(
                Path(file_path).name, pattern
            ):
                return True
        return False

    def _group_changes(self, files: List[str], group_by: str) -> Dict[str, List[str]]:
        """Group files by system or directory."""
        groups = {}

        if group_by == "system":
            for file_path in files:
                group = "other"
                for system, patterns in self.group_patterns.items():
                    if self._matches_pattern(file_path, patterns):
                        group = system
                        break

                if group not in groups:
                    groups[group] = []
                groups[group].append(file_path)

        elif group_by == "directory":
            for file_path in files:
                directory = str(Path(file_path).parent)
                if directory not in groups:
                    groups[directory] = []
                groups[directory].append(file_path)

        return groups

    def _extract_context_from_files(self, files: List[str]) -> Dict[str, str]:
        """Extract context from changed files."""
        context = {}

        # Analyze file paths for context
        model_files = [f for f in files if "model" in f.lower()]
        git_files = [f for f in files if "git" in f.lower()]
        core_files = [f for f in files if "core" in f.lower()]

        if model_files:
            context["system"] = "model interface"
            context["operation"] = "model operations"
        elif git_files:
            context["system"] = "git workflows"
            context["operation"] = "version control"
        elif core_files:
            context["system"] = "core functionality"
            context["operation"] = "system stability"
        else:
            context["system"] = "Mai"
            context["operation"] = "functionality"

        # Default scenario
        context["scenario"] = "your conversations"
        context["area"] = "Mai's capabilities"

        return context

    def _make_user_focused(self, message: str) -> str:
        """Convert message to be user-focused."""
        # Remove technical jargon
        replacements = {
            "feat:": "",
            "fix:": "",
            "refactor:": "",
            "optimize:": "",
            "implementation": "new capability",
            "functionality": "features",
            "module": "component",
            "code": "improvements",
            "api": "interface",
            "backend": "core system",
        }

        for old, new in replacements.items():
            message = message.replace(old, new)

        # Start with action verb if needed
        if not message[0].isupper():
            message = message[0].upper() + message[1:]

        return message.strip()

    def _validate_commit(self, message: str, files: Optional[List[str]]) -> Dict[str, Any]:
        """Validate commit before creation."""
        issues = []

        # Check message length
        if len(message) > 100:
            issues.append("Commit message too long (>100 characters)")

        # Check message has content
        if not message.strip():
            issues.append("Empty commit message")

        # Check for files if specified
        if files and not files:
            issues.append("No files specified for commit")

        return {"valid": len(issues) == 0, "issues": issues}

    def _extract_impact_from_message(self, message: str) -> str:
        """Extract impact description from commit message."""
        # Split by lines and take first non-empty line
        lines = message.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line and not line.startswith("Files:"):
                return line
        return message

    def _init_user_info(self) -> None:
        """Initialize user information from git config."""
        try:
            config = self.repo.config_reader()
            self.user_name = config.get_value("user", "name", "Mai")
            self.user_email = config.get_value("user", "email", "mai@local")
        except Exception:
            self.user_name = "Mai"
            self.user_email = "mai@local"

"""Secure sandbox execution interface for Python code."""

import logging
import tempfile
import time
import yaml
from typing import Dict, Any, Optional
from pathlib import Path

from .container_manager import ContainerManager

logger = logging.getLogger(__name__)


class SandboxExecutor:
    """Executes Python code securely in Docker containers."""

    def __init__(self, config_path: str = "config/sandbox.yaml"):
        """
        Initialize sandbox executor with configuration.

        Args:
            config_path: Path to sandbox configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.container_manager = ContainerManager()

    def _load_config(self) -> Dict[str, Any]:
        """Load sandbox configuration from YAML file."""
        try:
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {self.config_path} not found, using defaults")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default sandbox configuration."""
        return {
            "resources": {
                "cpu_count": 2,
                "mem_limit": "1g",
                "timeout": 120,
                "pids_limit": 100,
            },
            "security": {
                "read_only": True,
                "security_opt": ["no-new-privileges"],
                "cap_drop": ["ALL"],
                "user": "1000:1000",
            },
            "network": {"network_mode": "none"},
            "image": "python:3.11-slim",
        }

    def execute_code(self, code: str, trust_level: str = "trusted") -> Dict[str, Any]:
        """
        Execute Python code in a secure sandbox.

        Args:
            code: Python code to execute
            trust_level: Trust level affecting resource limits

        Returns:
            Execution result with output and metadata
        """
        start_time = time.time()
        container = None

        try:
            # Create execution environment
            runtime_configs = self._get_runtime_configs(trust_level)
            container = self.container_manager.create_container(
                image=self.config.get("image", "python:3.11-slim"),
                runtime_configs=runtime_configs,
            )

            # Prepare code file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                code_file = f.name

            # Copy code to container and execute
            self._copy_file_to_container(container, code_file, "/tmp/code.py")

            # Execute the code
            command = ["python3", "/tmp/code.py"]
            result = self.container_manager.run_command(
                container=container,
                command=command,
                timeout=runtime_configs.get("timeout", 120),
            )

            # Get resource statistics
            stats = self.container_manager.get_container_stats(container)

            execution_time = time.time() - start_time

            return {
                "success": result["exit_code"] == 0,
                "exit_code": result["exit_code"],
                "stdout": result["stdout"],
                "stderr": result["stderr"],
                "execution_time": round(execution_time, 3),
                "resource_usage": stats,
                "trust_level": trust_level,
                "container_id": container.id[:12] if container else None,
            }

        except Exception as e:
            logger.error(f"Sandbox execution failed: {e}")
            return {
                "success": False,
                "exit_code": -1,
                "stdout": "",
                "stderr": str(e),
                "execution_time": round(time.time() - start_time, 3),
                "resource_usage": {},
                "trust_level": trust_level,
                "container_id": container.id[:12] if container else None,
            }

        finally:
            # Cleanup
            if container:
                self.container_manager.cleanup_container(container)

    def _get_runtime_configs(self, trust_level: str) -> Dict[str, Any]:
        """
        Get runtime configuration based on trust level.

        Args:
            trust_level: Trust level affecting resource limits

        Returns:
            Runtime configuration dictionary
        """
        base_config = self.config.get("resources", {})
        security_config = self.config.get("security", {})
        network_config = self.config.get("network", {})

        # Adjust limits based on trust level
        if trust_level == "untrusted":
            base_config = {
                **base_config,
                "cpu_count": min(base_config.get("cpu_count", 2), 1),
                "mem_limit": "512m",
                "timeout": min(base_config.get("timeout", 120), 30),
                "pids_limit": 50,
            }
        elif trust_level == "trusted":
            # Use configured limits
            pass
        else:  # unknown trust level - use most restrictive
            base_config = {
                "cpu_count": 1,
                "mem_limit": "256m",
                "timeout": 15,
                "pids_limit": 25,
            }

        return {**base_config, **security_config, **network_config}

    def _copy_file_to_container(self, container, source_path: str, dest_path: str):
        """
        Copy file to container.

        Args:
            container: Docker container
            source_path: Source file path
            dest_path: Destination path in container
        """
        try:
            with open(source_path, "rb") as f:
                container.put_archive("/tmp", f)
            # Move file to correct location
            self.container_manager.run_command(
                container,
                ["mv", f"/tmp/{Path(source_path).name}", dest_path],
                timeout=5,
            )
        except Exception as e:
            logger.error(f"Failed to copy file to container: {e}")
            raise

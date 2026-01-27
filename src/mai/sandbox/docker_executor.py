"""
Docker Executor for Mai Safe Code Execution

Provides isolated container execution using Docker with comprehensive
resource limits, security restrictions, and audit logging integration.
"""

import logging
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import docker
    from docker.errors import APIError, ContainerError, DockerException, ImageNotFound
    from docker.models.containers import Container

    DOCKER_AVAILABLE = True
except ImportError:
    docker = None
    Container = None
    DockerException = Exception
    APIError = Exception
    ContainerError = Exception
    ImageNotFound = Exception
    DOCKER_AVAILABLE = False

from .audit_logger import AuditLogger


@dataclass
class ContainerConfig:
    """Configuration for Docker container execution"""

    image: str = "python:3.10-slim"
    timeout_seconds: int = 30
    memory_limit: str = "128m"  # Docker memory limit format
    cpu_limit: str = "0.5"  # CPU quota (0.5 = 50% of one CPU)
    network_disabled: bool = True
    read_only_filesystem: bool = True
    tmpfs_size: str = "64m"  # Temporary filesystem size
    working_dir: str = "/app"
    user: str = "nobody"  # Non-root user


@dataclass
class ContainerResult:
    """Result of container execution"""

    success: bool
    container_id: str
    exit_code: int
    stdout: str | None = None
    stderr: str | None = None
    execution_time: float = 0.0
    error: str | None = None
    resource_usage: dict[str, Any] | None = None


class DockerExecutor:
    """
    Docker-based container executor for isolated code execution.

    Provides secure sandboxing using Docker containers with resource limits,
    network restrictions, and comprehensive audit logging.
    """

    def __init__(self, audit_logger: AuditLogger | None = None):
        """
        Initialize Docker executor

        Args:
            audit_logger: Optional audit logger for execution logging
        """
        self.audit_logger = audit_logger
        self.client = None
        self.available = False

        # Try to initialize Docker client
        self._initialize_docker()

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def _initialize_docker(self) -> None:
        """Initialize Docker client and verify availability"""
        if not DOCKER_AVAILABLE:
            self.available = False
            return

        try:
            if docker is not None:
                self.client = docker.from_env()
                # Test Docker connection
                self.client.ping()
                self.available = True
            else:
                self.available = False
                self.client = None
        except Exception as e:
            self.logger.warning(f"Docker not available: {e}")
            self.available = False
            self.client = None

    def is_available(self) -> bool:
        """Check if Docker executor is available"""
        return self.available and self.client is not None

    def execute_code(
        self,
        code: str,
        config: ContainerConfig | None = None,
        environment: dict[str, str] | None = None,
        files: dict[str, str] | None = None,
    ) -> ContainerResult:
        """
        Execute code in isolated Docker container

        Args:
            code: Python code to execute
            config: Container configuration
            environment: Environment variables
            files: Additional files to mount in container

        Returns:
            ContainerResult with execution details
        """
        if not self.is_available() or self.client is None:
            return ContainerResult(
                success=False, container_id="", exit_code=-1, error="Docker executor not available"
            )

        config = config or ContainerConfig()
        container_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        try:
            # Create temporary directory for files
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Write code to file
                code_file = temp_path / "execute.py"
                code_file.write_text(code)

                # Prepare volume mounts
                volumes = {
                    str(code_file): {
                        "bind": f"{config.working_dir}/execute.py",
                        "mode": "ro",  # read-only
                    }
                }

                # Add additional files if provided
                if files:
                    for filename, content in files.items():
                        file_path = temp_path / filename
                        file_path.write_text(content)
                        volumes[str(file_path)] = {
                            "bind": f"{config.working_dir}/{filename}",
                            "mode": "ro",
                        }

                # Prepare container configuration
                container_config = self._build_container_config(config, environment)

                # Create and start container
                container = self.client.containers.run(
                    image=config.image,
                    command=["python", "execute.py"],
                    volumes=volumes,
                    **container_config,
                    detach=True,
                )

                # Get container ID safely
                container_id = getattr(container, "id", container_id)

                try:
                    # Wait for completion with timeout
                    result = container.wait(timeout=config.timeout_seconds)
                    exit_code = result["StatusCode"]

                    # Get logs
                    stdout = container.logs(stdout=True, stderr=False).decode("utf-8")
                    stderr = container.logs(stdout=False, stderr=True).decode("utf-8")

                    # Get resource usage stats
                    stats = self._get_container_stats(container)

                    # Determine success
                    success = exit_code == 0 and not stderr

                    execution_result = ContainerResult(
                        success=success,
                        container_id=container_id,
                        exit_code=exit_code,
                        stdout=stdout,
                        stderr=stderr,
                        execution_time=time.time() - start_time,
                        resource_usage=stats,
                    )

                    # Log execution if audit logger available
                    if self.audit_logger:
                        self._log_container_execution(code, execution_result, config)

                    return execution_result

                finally:
                    # Always cleanup container
                    try:
                        container.remove(force=True)
                    except Exception:
                        pass  # Best effort cleanup

        except ContainerError as e:
            return ContainerResult(
                success=False,
                container_id=container_id or "unknown",
                exit_code=getattr(e, "exit_code", -1),
                stderr=str(e),
                execution_time=time.time() - start_time,
                error=f"Container execution error: {e}",
            )

        except ImageNotFound as e:
            return ContainerResult(
                success=False,
                container_id=container_id,
                exit_code=-1,
                error=f"Docker image not found: {e}",
            )

        except APIError as e:
            return ContainerResult(
                success=False,
                container_id=container_id,
                exit_code=-1,
                error=f"Docker API error: {e}",
            )

        except Exception as e:
            return ContainerResult(
                success=False,
                container_id=container_id,
                exit_code=-1,
                execution_time=time.time() - start_time,
                error=f"Unexpected error: {e}",
            )

    def _build_container_config(
        self, config: ContainerConfig, environment: dict[str, str] | None = None
    ) -> dict[str, Any]:
        """Build Docker container configuration"""
        container_config = {
            "mem_limit": config.memory_limit,
            "cpu_quota": int(float(config.cpu_limit) * 100000),  # Convert to microseconds
            "cpu_period": 100000,  # 100ms period
            "network_disabled": config.network_disabled,
            "read_only": config.read_only_filesystem,
            "tmpfs": {"/tmp": f"size={config.tmpfs_size},noexec,nosuid,nodev"},
            "user": config.user,
            "working_dir": config.working_dir,
            "remove": True,  # Auto-remove container
        }

        # Add environment variables
        if environment:
            container_config["environment"] = {
                **environment,
                "PYTHONPATH": config.working_dir,
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        else:
            container_config["environment"] = {
                "PYTHONPATH": config.working_dir,
                "PYTHONDONTWRITEBYTECODE": "1",
            }

        # Security options
        container_config["security_opt"] = [
            "no-new-privileges:true",
            "seccomp:unconfined",  # Python needs some syscalls
        ]

        # Capabilities (drop all capabilities)
        container_config["cap_drop"] = ["ALL"]
        container_config["cap_add"] = ["CHOWN", "DAC_OVERRIDE"]  # Minimal capabilities for Python

        return container_config

    def _get_container_stats(self, container) -> dict[str, Any]:
        """Get resource usage statistics from container"""
        try:
            stats = container.stats(stream=False)

            # Calculate CPU usage
            cpu_stats = stats.get("cpu_stats", {})
            precpu_stats = stats.get("precpu_stats", {})

            cpu_usage = cpu_stats.get("cpu_usage", {}).get("total_usage", 0)
            precpu_usage = precpu_stats.get("cpu_usage", {}).get("total_usage", 0)

            system_usage = cpu_stats.get("system_cpu_usage", 0)
            presystem_usage = precpu_stats.get("system_cpu_usage", 0)

            cpu_count = cpu_stats.get("online_cpus", 1)

            cpu_percent = 0.0
            if system_usage > presystem_usage:
                cpu_delta = cpu_usage - precpu_usage
                system_delta = system_usage - presystem_usage
                cpu_percent = (cpu_delta / system_delta) * cpu_count * 100.0

            # Calculate memory usage
            memory_stats = stats.get("memory_stats", {})
            memory_usage = memory_stats.get("usage", 0)
            memory_limit = memory_stats.get("limit", 1)
            memory_percent = (memory_usage / memory_limit) * 100.0

            return {
                "cpu_percent": round(cpu_percent, 2),
                "memory_usage_bytes": memory_usage,
                "memory_limit_bytes": memory_limit,
                "memory_percent": round(memory_percent, 2),
                "memory_usage_mb": round(memory_usage / (1024 * 1024), 2),
            }

        except Exception:
            return {
                "cpu_percent": 0.0,
                "memory_usage_bytes": 0,
                "memory_limit_bytes": 0,
                "memory_percent": 0.0,
                "memory_usage_mb": 0.0,
            }

    def _log_container_execution(
        self, code: str, result: ContainerResult, config: ContainerConfig
    ) -> None:
        """Log container execution to audit logger"""
        if not self.audit_logger:
            return

        execution_data = {
            "type": "docker_container",
            "container_id": result.container_id,
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "execution_time": result.execution_time,
            "config": {
                "image": config.image,
                "timeout": config.timeout_seconds,
                "memory_limit": config.memory_limit,
                "cpu_limit": config.cpu_limit,
                "network_disabled": config.network_disabled,
                "read_only_filesystem": config.read_only_filesystem,
            },
            "resource_usage": result.resource_usage,
        }

        # Note: execution_type parameter not available in current AuditLogger
        self.audit_logger.log_execution(code=code, execution_result=execution_data)

    def get_available_images(self) -> list[str]:
        """Get list of available Docker images"""
        if not self.is_available() or self.client is None:
            return []

        try:
            images = self.client.images.list()
            return [img.tags[0] for img in images if img.tags]
        except Exception:
            return []

    def pull_image(self, image_name: str) -> bool:
        """Pull Docker image"""
        if not self.is_available() or self.client is None:
            return False

        try:
            self.client.images.pull(image_name)
            return True
        except Exception:
            return False

    def cleanup_containers(self) -> int:
        """Clean up any dangling containers"""
        if not self.is_available() or self.client is None:
            return 0

        try:
            containers = self.client.containers.list(all=True, filters={"status": "exited"})
            count = 0
            for container in containers:
                try:
                    container.remove(force=True)
                    count += 1
                except Exception:
                    pass
            return count
        except Exception:
            return 0

    def get_system_info(self) -> dict[str, Any]:
        """Get Docker system information"""
        if not self.is_available() or self.client is None:
            return {"available": False}

        try:
            info = self.client.info()
            version = self.client.version()

            return {
                "available": True,
                "version": version.get("Version", "unknown"),
                "api_version": version.get("ApiVersion", "unknown"),
                "containers": info.get("Containers", 0),
                "containers_running": info.get("ContainersRunning", 0),
                "containers_paused": info.get("ContainersPaused", 0),
                "containers_stopped": info.get("ContainersStopped", 0),
                "images": info.get("Images", 0),
                "memory_total": info.get("MemTotal", 0),
                "ncpu": info.get("NCPU", 0),
            }
        except Exception:
            return {"available": False, "error": "Failed to get system info"}

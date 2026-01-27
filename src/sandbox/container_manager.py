"""Docker container management for secure sandbox execution."""

import docker
import logging
from typing import Dict, List, Optional, Any
from docker.models.containers import Container

logger = logging.getLogger(__name__)


class ContainerManager:
    """Manages Docker containers with security hardening for sandbox execution."""

    def __init__(self):
        """Initialize Docker client."""
        try:
            self.client = docker.from_env()
            self.client.ping()
        except Exception as e:
            logger.error(f"Failed to connect to Docker: {e}")
            raise

    def create_container(
        self,
        image: str = "python:3.11-slim",
        runtime_configs: Optional[Dict[str, Any]] = None,
    ) -> Container:
        """
        Create a secure Docker container with security hardening.

        Args:
            image: Docker image to use
            runtime_configs: Additional runtime configuration

        Returns:
            Created container object
        """
        configs = runtime_configs or {}

        # Security hardening defaults
        security_opts = configs.get("security_opt", ["no-new-privileges"])
        cap_drop = configs.get("cap_drop", ["ALL"])
        read_only = configs.get("read_only", True)
        user = configs.get("user", "1000:1000")  # Non-root user
        network_mode = configs.get("network_mode", "none")

        # Resource limits
        mem_limit = configs.get("mem_limit", "1g")
        cpu_count = configs.get("cpu_count", 2)
        pids_limit = configs.get("pids_limit", 100)

        container = self.client.containers.create(
            image=image,
            security_opt=security_opts,
            cap_drop=cap_drop,
            read_only=read_only,
            user=user,
            network_mode=network_mode,
            mem_limit=mem_limit,
            cpu_count=cpu_count,
            pids_limit=pids_limit,
            detach=True,
            remove=False,
        )

        logger.info(f"Created secure container {container.id[:12]}")
        return container

    def run_command(
        self, container: Container, command: List[str], timeout: Optional[int] = 120
    ) -> Dict[str, Any]:
        """
        Execute command in container with timeout.

        Args:
            container: Container to run command in
            command: Command to execute
            timeout: Timeout in seconds

        Returns:
            Execution result with output and metadata
        """
        try:
            # Start container if not running
            if container.status != "running":
                container.start()

            # Execute command
            result = container.exec_run(cmd=command, timeout=timeout, demux=True)

            return {
                "exit_code": result.exit_code,
                "stdout": result.output[0].decode("utf-8") if result.output[0] else "",
                "stderr": result.output[1].decode("utf-8") if result.output[1] else "",
                "exec_id": result.id,
            }

        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return {"exit_code": -1, "stdout": "", "stderr": str(e), "exec_id": None}

    def get_container_stats(self, container: Container) -> Dict[str, Any]:
        """
        Get real-time resource statistics for container.

        Args:
            container: Container to get stats for

        Returns:
            Resource usage statistics
        """
        try:
            stats = container.stats(stream=False)

            # CPU usage calculation
            cpu_delta = (
                stats["cpu_stats"]["cpu_usage"]["total_usage"]
                - stats["precpu_stats"]["cpu_usage"]["total_usage"]
            )
            system_cpu_delta = (
                stats["cpu_stats"]["system_cpu_usage"]
                - stats["precpu_stats"]["system_cpu_usage"]
            )
            cpu_count = len(stats["cpu_stats"]["cpu_usage"]["percpu_usage"])
            cpu_percent = (cpu_delta / system_cpu_delta) * cpu_count * 100.0

            # Memory usage
            memory_usage = stats["memory_stats"]["usage"]
            memory_limit = stats["memory_stats"]["limit"]
            memory_percent = (memory_usage / memory_limit) * 100.0

            return {
                "cpu_percent": round(cpu_percent, 2),
                "memory_usage_mb": round(memory_usage / (1024 * 1024), 2),
                "memory_limit_mb": round(memory_limit / (1024 * 1024), 2),
                "memory_percent": round(memory_percent, 2),
                "pids_current": stats.get("pids_stats", {}).get("current", 0),
            }

        except Exception as e:
            logger.error(f"Failed to get container stats: {e}")
            return {}

    def cleanup_container(self, container: Container) -> bool:
        """
        Clean up container by stopping and removing it.

        Args:
            container: Container to cleanup

        Returns:
            True if cleanup successful
        """
        try:
            if container.status == "running":
                container.stop(timeout=5)
            container.remove(force=True)
            logger.info(f"Cleaned up container {container.id[:12]}")
            return True
        except Exception as e:
            logger.error(f"Failed to cleanup container {container.id[:12]}: {e}")
            return False

    def list_containers(self, all_containers: bool = False) -> List[Container]:
        """
        List containers managed by this manager.

        Args:
            all_containers: Include stopped containers

        Returns:
            List of containers
        """
        return self.client.containers.list(all=all_containers)

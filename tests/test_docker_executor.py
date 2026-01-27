"""
Tests for Docker Executor component

Test suite for Docker-based container execution with isolation,
resource limits, and audit logging integration.
"""

import pytest
import tempfile
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Import components under test
from src.mai.sandbox.docker_executor import DockerExecutor, ContainerConfig, ContainerResult
from src.mai.sandbox.audit_logger import AuditLogger


class TestContainerConfig:
    """Test ContainerConfig dataclass"""

    def test_default_config(self):
        """Test default configuration values"""
        config = ContainerConfig()
        assert config.image == "python:3.10-slim"
        assert config.timeout_seconds == 30
        assert config.memory_limit == "128m"
        assert config.cpu_limit == "0.5"
        assert config.network_disabled is True
        assert config.read_only_filesystem is True
        assert config.tmpfs_size == "64m"
        assert config.working_dir == "/app"
        assert config.user == "nobody"

    def test_custom_config(self):
        """Test custom configuration values"""
        config = ContainerConfig(
            image="python:3.9-alpine",
            timeout_seconds=60,
            memory_limit="256m",
            cpu_limit="0.8",
            network_disabled=False,
        )
        assert config.image == "python:3.9-alpine"
        assert config.timeout_seconds == 60
        assert config.memory_limit == "256m"
        assert config.cpu_limit == "0.8"
        assert config.network_disabled is False


class TestDockerExecutor:
    """Test DockerExecutor class"""

    @pytest.fixture
    def mock_audit_logger(self):
        """Create mock audit logger"""
        return Mock(spec=AuditLogger)

    @pytest.fixture
    def docker_executor(self, mock_audit_logger):
        """Create DockerExecutor instance for testing"""
        return DockerExecutor(audit_logger=mock_audit_logger)

    def test_init_without_docker(self, mock_audit_logger):
        """Test initialization when Docker is not available"""
        with patch("src.mai.sandbox.docker_executor.DOCKER_AVAILABLE", False):
            executor = DockerExecutor(audit_logger=mock_audit_logger)
            assert executor.is_available() is False
            assert executor.client is None

    def test_init_with_docker_error(self, mock_audit_logger):
        """Test initialization when Docker fails to connect"""
        with patch("src.mai.sandbox.docker_executor.DOCKER_AVAILABLE", True):
            with patch("docker.from_env") as mock_from_env:
                mock_from_env.side_effect = Exception("Docker daemon not running")

                executor = DockerExecutor(audit_logger=mock_audit_logger)
                assert executor.is_available() is False
                assert executor.client is None

    def test_is_available(self, docker_executor):
        """Test is_available method"""
        # When client is None, should not be available
        docker_executor.client = None
        docker_executor.available = False
        assert docker_executor.is_available() is False

        # When client is available, should reflect available status
        docker_executor.client = Mock()
        docker_executor.available = True
        assert docker_executor.is_available() is True

        docker_executor.client = Mock()
        docker_executor.available = False
        assert docker_executor.is_available() is False

    def test_execute_code_unavailable(self, docker_executor):
        """Test execute_code when Docker is not available"""
        with patch.object(docker_executor, "is_available", return_value=False):
            result = docker_executor.execute_code("print('test')")

            assert result.success is False
            assert result.container_id == ""
            assert result.exit_code == -1
            assert "Docker executor not available" in result.error

    @patch("src.mai.sandbox.docker_executor.Path")
    @patch("src.mai.sandbox.docker_executor.tempfile.TemporaryDirectory")
    def test_execute_code_success(self, mock_temp_dir, mock_path, docker_executor):
        """Test successful code execution in container"""
        # Mock temporary directory and file creation
        mock_temp_file = Mock()
        mock_temp_file.write_text = Mock()

        mock_temp_path = Mock()
        mock_temp_path.__truediv__ = Mock(return_value=mock_temp_file)
        mock_temp_path.__str__ = Mock(return_value="/tmp/test")

        mock_temp_dir.return_value.__enter__.return_value = mock_temp_path

        # Mock Docker client and container
        mock_container = Mock()
        mock_container.id = "test-container-id"
        mock_container.wait.return_value = {"StatusCode": 0}
        mock_container.logs.return_value = b"test output"
        mock_container.stats.return_value = {
            "cpu_stats": {"cpu_usage": {"total_usage": 1000000}, "system_cpu_usage": 2000000},
            "precpu_stats": {"cpu_usage": {"total_usage": 500000}, "system_cpu_usage": 1000000},
            "memory_stats": {"usage": 50000000, "limit": 100000000},
        }

        mock_client = Mock()
        mock_client.containers.run.return_value = mock_container

        docker_executor.client = mock_client
        docker_executor.available = True

        # Execute code
        result = docker_executor.execute_code("print('test')")

        assert result.success is True
        assert result.container_id == "test-container-id"
        assert result.exit_code == 0
        assert result.stdout == "test output"
        assert result.execution_time > 0
        assert result.resource_usage is not None

    @patch("src.mai.sandbox.docker_executor.Path")
    @patch("src.mai.sandbox.docker_executor.tempfile.TemporaryDirectory")
    def test_execute_code_with_files(self, mock_temp_dir, mock_path, docker_executor):
        """Test code execution with additional files"""
        # Mock temporary directory and file creation
        mock_temp_file = Mock()
        mock_temp_file.write_text = Mock()

        mock_temp_path = Mock()
        mock_temp_path.__truediv__ = Mock(return_value=mock_temp_file)
        mock_temp_path.__str__ = Mock(return_value="/tmp/test")

        mock_temp_dir.return_value.__enter__.return_value = mock_temp_path

        # Mock Docker client and container
        mock_container = Mock()
        mock_container.id = "test-container-id"
        mock_container.wait.return_value = {"StatusCode": 0}
        mock_container.logs.return_value = b"test output"
        mock_container.stats.return_value = {}

        mock_client = Mock()
        mock_client.containers.run.return_value = mock_container

        docker_executor.client = mock_client
        docker_executor.available = True

        # Execute code with files
        files = {"data.txt": "test data"}
        result = docker_executor.execute_code("print('test')", files=files)

        # Verify additional files were handled
        assert mock_temp_file.write_text.call_count >= 2  # code + data file
        assert result.success is True

    def test_build_container_config(self, docker_executor):
        """Test building Docker container configuration"""
        config = ContainerConfig(memory_limit="256m", cpu_limit="0.8", network_disabled=False)
        environment = {"TEST_VAR": "test_value"}

        container_config = docker_executor._build_container_config(config, environment)

        assert container_config["mem_limit"] == "256m"
        assert container_config["cpu_quota"] == 80000  # 0.8 * 100000
        assert container_config["cpu_period"] == 100000
        assert container_config["network_disabled"] is False
        assert container_config["read_only"] is True
        assert container_config["user"] == "nobody"
        assert container_config["working_dir"] == "/app"
        assert "TEST_VAR" in container_config["environment"]
        assert "security_opt" in container_config
        assert "cap_drop" in container_config
        assert "cap_add" in container_config

    def test_get_container_stats(self, docker_executor):
        """Test extracting container resource statistics"""
        mock_container = Mock()
        mock_container.stats.return_value = {
            "cpu_stats": {
                "cpu_usage": {"total_usage": 2000000},
                "system_cpu_usage": 4000000,
                "online_cpus": 2,
            },
            "precpu_stats": {"cpu_usage": {"total_usage": 1000000}, "system_cpu_usage": 2000000},
            "memory_stats": {
                "usage": 67108864,  # 64MB
                "limit": 134217728,  # 128MB
            },
        }

        stats = docker_executor._get_container_stats(mock_container)

        assert stats["cpu_percent"] == 100.0  # (2000000-1000000)/(4000000-2000000) * 2 * 100
        assert stats["memory_usage_bytes"] == 67108864
        assert stats["memory_limit_bytes"] == 134217728
        assert stats["memory_percent"] == 50.0
        assert stats["memory_usage_mb"] == 64.0

    def test_get_container_stats_error(self, docker_executor):
        """Test get_container_stats with error"""
        mock_container = Mock()
        mock_container.stats.side_effect = Exception("Stats error")

        stats = docker_executor._get_container_stats(mock_container)

        assert stats["cpu_percent"] == 0.0
        assert stats["memory_usage_bytes"] == 0
        assert stats["memory_percent"] == 0.0
        assert stats["memory_usage_mb"] == 0.0

    def test_log_container_execution(self, docker_executor, mock_audit_logger):
        """Test logging container execution"""
        config = ContainerConfig(image="python:3.10-slim")
        result = ContainerResult(
            success=True,
            container_id="test-id",
            exit_code=0,
            stdout="test output",
            stderr="",
            execution_time=1.5,
            resource_usage={"cpu_percent": 50.0},
        )

        docker_executor._log_container_execution("print('test')", result, config)

        # Verify audit logger was called
        mock_audit_logger.log_execution.assert_called_once()
        call_args = mock_audit_logger.log_execution.call_args
        assert call_args.kwargs["code"] == "print('test')"
        assert call_args.kwargs["execution_type"] == "docker"
        assert "docker_container" in call_args.kwargs["execution_result"]["type"]

    def test_get_available_images(self, docker_executor):
        """Test getting available Docker images"""
        mock_image = Mock()
        mock_image.tags = ["python:3.10-slim", "python:3.9-alpine"]

        mock_client = Mock()
        mock_client.images.list.return_value = [mock_image]

        docker_executor.client = mock_client
        docker_executor.available = True

        images = docker_executor.get_available_images()

        assert "python:3.10-slim" in images
        assert "python:3.9-alpine" in images

    def test_pull_image(self, docker_executor):
        """Test pulling Docker image"""
        mock_client = Mock()
        mock_client.images.pull.return_value = None

        docker_executor.client = mock_client
        docker_executor.available = True

        result = docker_executor.pull_image("python:3.10-slim")

        assert result is True
        mock_client.images.pull.assert_called_once_with("python:3.10-slim")

    def test_cleanup_containers(self, docker_executor):
        """Test cleaning up containers"""
        mock_container = Mock()

        mock_client = Mock()
        mock_client.containers.list.return_value = [mock_container, mock_container]

        docker_executor.client = mock_client
        docker_executor.available = True

        count = docker_executor.cleanup_containers()

        assert count == 2
        assert mock_container.remove.call_count == 2

    def test_get_system_info(self, docker_executor):
        """Test getting Docker system information"""
        mock_client = Mock()
        mock_client.info.return_value = {
            "Containers": 5,
            "ContainersRunning": 2,
            "Images": 10,
            "MemTotal": 8589934592,
            "NCPU": 4,
        }
        mock_client.version.return_value = {"Version": "20.10.7", "ApiVersion": "1.41"}

        docker_executor.client = mock_client
        docker_executor.available = True

        info = docker_executor.get_system_info()

        assert info["available"] is True
        assert info["version"] == "20.10.7"
        assert info["api_version"] == "1.41"
        assert info["containers"] == 5
        assert info["images"] == 10


class TestDockerExecutorIntegration:
    """Integration tests for Docker executor with other sandbox components"""

    @pytest.fixture
    def mock_audit_logger(self):
        """Create mock audit logger"""
        return Mock(spec=AuditLogger)

    def test_docker_executor_integration(self, mock_audit_logger):
        """Test Docker executor integration with audit logger"""
        executor = DockerExecutor(audit_logger=mock_audit_logger)

        # Test that audit logger is properly integrated
        assert executor.audit_logger is mock_audit_logger

        # Mock Docker availability for integration test
        with patch.object(executor, "is_available", return_value=False):
            result = executor.execute_code("print('test')")

            # Should fail gracefully and still attempt logging
            assert result.success is False

    def test_container_result_serialization(self):
        """Test ContainerResult can be properly serialized"""
        result = ContainerResult(
            success=True,
            container_id="test-id",
            exit_code=0,
            stdout="test output",
            stderr="",
            execution_time=1.5,
            resource_usage={"cpu_percent": 50.0},
        )

        # Test that result can be converted to dict for JSON serialization
        result_dict = {
            "success": result.success,
            "container_id": result.container_id,
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "execution_time": result.execution_time,
            "error": result.error,
            "resource_usage": result.resource_usage,
        }

        assert result_dict["success"] is True
        assert result_dict["container_id"] == "test-id"


if __name__ == "__main__":
    pytest.main([__file__])

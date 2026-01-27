"""
Integration test for complete Docker sandbox execution

Tests the full integration of Docker executor with sandbox manager,
risk analysis, resource enforcement, and audit logging.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, Mock

from src.mai.sandbox.manager import SandboxManager, ExecutionRequest
from src.mai.sandbox.audit_logger import AuditLogger


@pytest.mark.integration
class TestDockerSandboxIntegration:
    """Integration tests for Docker sandbox execution"""

    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary directory for audit logs"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def sandbox_manager(self, temp_log_dir):
        """Create SandboxManager with temp log directory"""
        return SandboxManager(log_dir=temp_log_dir)

    def test_full_docker_execution_workflow(self, sandbox_manager):
        """Test complete Docker execution workflow"""
        with patch.object(sandbox_manager.docker_executor, "is_available", return_value=True):
            with patch.object(sandbox_manager.docker_executor, "execute_code") as mock_docker:
                # Mock Docker container execution
                from src.mai.sandbox.docker_executor import ContainerResult

                mock_docker.return_value = {
                    "success": True,
                    "output": "42\n",
                    "container_result": ContainerResult(
                        success=True,
                        container_id="integration-test-container",
                        exit_code=0,
                        stdout="42\n",
                        stderr="",
                        execution_time=2.3,
                        resource_usage={
                            "cpu_percent": 15.2,
                            "memory_usage_mb": 28.5,
                            "memory_percent": 5.5,
                        },
                    ),
                }

                # Create execution request
                request = ExecutionRequest(
                    code="result = 6 * 7\nprint(result)",
                    use_docker=True,
                    docker_image="python:3.10-slim",
                    timeout_seconds=30,
                    cpu_limit_percent=50.0,
                    memory_limit_percent=40.0,
                    network_allowed=False,
                    filesystem_restricted=True,
                )

                # Execute code
                result = sandbox_manager.execute_code(request)

                # Verify execution results
                assert result.success is True
                assert result.execution_method == "docker"
                assert result.output == "42\n"
                assert result.container_result is not None
                assert result.container_result.container_id == "integration-test-container"
                assert result.container_result.exit_code == 0
                assert result.container_result.execution_time == 2.3
                assert result.container_result.resource_usage["cpu_percent"] == 15.2
                assert result.container_result.resource_usage["memory_usage_mb"] == 28.5

                # Verify Docker executor was called with correct parameters
                mock_docker.assert_called_once()
                call_args = mock_docker.call_args

                # Check code was passed correctly
                assert call_args.args[0] == "result = 6 * 7\nprint(result)"

                # Check container config
                config = call_args.kwargs["config"]
                assert config.image == "python:3.10-slim"
                assert config.timeout_seconds == 30
                assert config.memory_limit == "51m"  # Scaled from 40% of 128m
                assert config.cpu_limit == "0.5"  # 50% CPU
                assert config.network_disabled is True
                assert config.read_only_filesystem is True

                # Verify audit logging occurred
                assert result.audit_entry_id is not None

                # Check audit log contents
                logs = sandbox_manager.get_execution_history(limit=1)
                assert len(logs) == 1

                log_entry = logs[0]
                assert log_entry["code"] == "result = 6 * 7\nprint(result)"
                assert log_entry["execution_result"]["success"] is True
                assert "docker_container" in log_entry["execution_result"]

    def test_docker_execution_with_additional_files(self, sandbox_manager):
        """Test Docker execution with additional files"""
        with patch.object(sandbox_manager.docker_executor, "is_available", return_value=True):
            with patch.object(sandbox_manager.docker_executor, "execute_code") as mock_docker:
                # Mock Docker execution
                from src.mai.sandbox.docker_executor import ContainerResult

                mock_docker.return_value = {
                    "success": True,
                    "output": "Hello, Alice!\n",
                    "container_result": ContainerResult(
                        success=True,
                        container_id="files-test-container",
                        exit_code=0,
                        stdout="Hello, Alice!\n",
                    ),
                }

                # Create execution request with additional files
                request = ExecutionRequest(
                    code="with open('template.txt', 'r') as f: template = f.read()\nprint(template.replace('{name}', 'Alice'))",
                    use_docker=True,
                    additional_files={"template.txt": "Hello, {name}!"},
                )

                # Execute code
                result = sandbox_manager.execute_code(request)

                # Verify execution
                assert result.success is True
                assert result.execution_method == "docker"

                # Verify Docker executor was called with files
                call_args = mock_docker.call_args
                assert "files" in call_args.kwargs
                assert call_args.kwargs["files"] == {"template.txt": "Hello, {name}!"}

    def test_docker_execution_blocked_by_risk_analysis(self, sandbox_manager):
        """Test that high-risk code is blocked before Docker execution"""
        with patch.object(sandbox_manager.docker_executor, "is_available", return_value=True):
            with patch.object(sandbox_manager.docker_executor, "execute_code") as mock_docker:
                # Risk analysis will automatically detect the dangerous pattern
                request = ExecutionRequest(
                    code="import subprocess; subprocess.run(['rm', '-rf', '/'], shell=True)",
                    use_docker=True,
                )

                # Execute code
                result = sandbox_manager.execute_code(request)

                # Verify execution was blocked
                assert result.success is False
                assert "blocked" in result.error.lower()
                assert result.risk_assessment.score >= 70
                assert result.execution_method == "local"  # Set before Docker check

                # Docker executor should not be called
                mock_docker.assert_not_called()

                # Should still be logged
                assert result.audit_entry_id is not None

    def test_docker_execution_fallback_to_local(self, sandbox_manager):
        """Test fallback to local execution when Docker unavailable"""
        with patch.object(sandbox_manager.docker_executor, "is_available", return_value=False):
            with patch.object(sandbox_manager, "_execute_in_sandbox") as mock_local:
                with patch.object(
                    sandbox_manager.resource_enforcer, "stop_monitoring"
                ) as mock_monitoring:
                    # Mock local execution
                    mock_local.return_value = {"success": True, "output": "Local fallback result"}

                    # Mock resource usage
                    from src.mai.sandbox.resource_enforcer import ResourceUsage

                    mock_monitoring.return_value = ResourceUsage(
                        cpu_percent=35.0,
                        memory_percent=25.0,
                        memory_used_gb=0.4,
                        elapsed_seconds=1.8,
                        approaching_limits=False,
                    )

                    # Create request preferring Docker
                    request = ExecutionRequest(
                        code="print('fallback test')",
                        use_docker=True,  # But Docker is unavailable
                    )

                    # Execute code
                    result = sandbox_manager.execute_code(request)

                    # Verify fallback to local execution
                    assert result.success is True
                    assert result.execution_method == "local"
                    assert result.output == "Local fallback result"
                    assert result.container_result is None
                    assert result.resource_usage is not None
                    assert result.resource_usage.cpu_percent == 35.0

                    # Verify local execution was used
                    mock_local.assert_called_once()

    def test_audit_logging_docker_execution_details(self, sandbox_manager):
        """Test comprehensive audit logging for Docker execution"""
        with patch.object(sandbox_manager.docker_executor, "is_available", return_value=True):
            with patch.object(sandbox_manager.docker_executor, "execute_code") as mock_docker:
                # Mock Docker execution with detailed stats
                from src.mai.sandbox.docker_executor import ContainerResult

                mock_docker.return_value = {
                    "success": True,
                    "output": "Calculation complete: 144\n",
                    "container_result": ContainerResult(
                        success=True,
                        container_id="audit-test-container",
                        exit_code=0,
                        stdout="Calculation complete: 144\n",
                        stderr="",
                        execution_time=3.7,
                        resource_usage={
                            "cpu_percent": 22.8,
                            "memory_usage_mb": 45.2,
                            "memory_percent": 8.9,
                            "memory_usage_bytes": 47395648,
                            "memory_limit_bytes": 536870912,
                        },
                    ),
                }

                # Execute request
                request = ExecutionRequest(
                    code="result = 12 * 12\nprint(f'Calculation complete: {result}')",
                    use_docker=True,
                    docker_image="python:3.9-alpine",
                    timeout_seconds=45,
                )

                result = sandbox_manager.execute_code(request)

                # Verify audit log contains Docker execution details
                logs = sandbox_manager.get_execution_history(limit=1)
                assert len(logs) == 1

                log_entry = logs[0]
                execution_result = log_entry["execution_result"]

                # Check Docker-specific fields
                assert execution_result["type"] == "docker_container"
                assert execution_result["container_id"] == "audit-test-container"
                assert execution_result["exit_code"] == 0
                assert execution_result["stdout"] == "Calculation complete: 144\n"

                # Check configuration details
                config = execution_result["config"]
                assert config["image"] == "python:3.9-alpine"
                assert config["timeout"] == 45
                assert config["network_disabled"] is True
                assert config["read_only_filesystem"] is True

                # Check resource usage
                resource_usage = execution_result["resource_usage"]
                assert resource_usage["cpu_percent"] == 22.8
                assert resource_usage["memory_usage_mb"] == 45.2
                assert resource_usage["memory_percent"] == 8.9

    def test_system_status_includes_docker_info(self, sandbox_manager):
        """Test system status includes Docker information"""
        with patch.object(sandbox_manager.docker_executor, "is_available", return_value=True):
            with patch.object(
                sandbox_manager.docker_executor, "get_system_info"
            ) as mock_docker_info:
                # Mock Docker system info
                mock_docker_info.return_value = {
                    "available": True,
                    "version": "20.10.12",
                    "api_version": "1.41",
                    "containers": 5,
                    "containers_running": 2,
                    "images": 8,
                    "ncpu": 4,
                    "memory_total": 8589934592,
                }

                # Get system status
                status = sandbox_manager.get_system_status()

                # Verify Docker information is included
                assert "docker_available" in status
                assert "docker_info" in status
                assert status["docker_available"] is True
                assert status["docker_info"]["available"] is True
                assert status["docker_info"]["version"] == "20.10.12"
                assert status["docker_info"]["containers"] == 5
                assert status["docker_info"]["images"] == 8

    def test_docker_status_management(self, sandbox_manager):
        """Test Docker status management functions"""
        with patch.object(sandbox_manager.docker_executor, "is_available", return_value=True):
            with patch.object(
                sandbox_manager.docker_executor, "get_available_images"
            ) as mock_images:
                with patch.object(sandbox_manager.docker_executor, "pull_image") as mock_pull:
                    with patch.object(
                        sandbox_manager.docker_executor, "cleanup_containers"
                    ) as mock_cleanup:
                        # Mock responses
                        mock_images.return_value = ["python:3.10-slim", "python:3.9-alpine"]
                        mock_pull.return_value = True
                        mock_cleanup.return_value = 3

                        # Test get Docker status
                        status = sandbox_manager.get_docker_status()
                        assert status["available"] is True
                        assert "python:3.10-slim" in status["images"]
                        assert "python:3.9-alpine" in status["images"]

                        # Test pull image
                        pull_result = sandbox_manager.pull_docker_image("node:16-alpine")
                        assert pull_result is True
                        mock_pull.assert_called_once_with("node:16-alpine")

                        # Test cleanup containers
                        cleanup_count = sandbox_manager.cleanup_docker_containers()
                        assert cleanup_count == 3
                        mock_cleanup.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

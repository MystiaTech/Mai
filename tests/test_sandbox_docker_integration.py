"""
Tests for SandboxManager with Docker integration

Test suite for enhanced SandboxManager that includes Docker-based
container execution with fallback to local execution.
"""

import pytest
from unittest.mock import Mock, patch, call

from src.mai.sandbox.manager import SandboxManager, ExecutionRequest, ExecutionResult
from src.mai.sandbox.risk_analyzer import RiskAssessment, RiskPattern
from src.mai.sandbox.resource_enforcer import ResourceUsage, ResourceLimits
from src.mai.sandbox.docker_executor import ContainerResult, ContainerConfig


class TestSandboxManagerDockerIntegration:
    """Test SandboxManager Docker integration features"""

    @pytest.fixture
    def sandbox_manager(self):
        """Create SandboxManager instance for testing"""
        return SandboxManager()

    @pytest.fixture
    def mock_docker_executor(self):
        """Create mock Docker executor"""
        mock_executor = Mock()
        mock_executor.is_available.return_value = True
        mock_executor.execute_code.return_value = ContainerResult(
            success=True,
            container_id="test-container-id",
            exit_code=0,
            stdout="Hello from Docker!",
            stderr="",
            execution_time=1.2,
            resource_usage={"cpu_percent": 45.0, "memory_usage_mb": 32.0},
        )
        mock_executor.get_system_info.return_value = {
            "available": True,
            "version": "20.10.7",
            "containers": 3,
        }
        return mock_executor

    def test_execution_request_with_docker_options(self):
        """Test ExecutionRequest with Docker-specific options"""
        request = ExecutionRequest(
            code="print('test')",
            use_docker=True,
            docker_image="python:3.9-alpine",
            timeout_seconds=45,
            network_allowed=True,
            additional_files={"data.txt": "test content"},
        )

        assert request.use_docker is True
        assert request.docker_image == "python:3.9-alpine"
        assert request.timeout_seconds == 45
        assert request.network_allowed is True
        assert request.additional_files == {"data.txt": "test content"}

    def test_execution_result_with_docker_info(self):
        """Test ExecutionResult includes Docker execution info"""
        container_result = ContainerResult(
            success=True,
            container_id="test-id",
            exit_code=0,
            stdout="Docker output",
            execution_time=1.5,
        )

        result = ExecutionResult(
            success=True,
            execution_id="test-exec",
            output="Docker output",
            execution_method="docker",
            container_result=container_result,
        )

        assert result.execution_method == "docker"
        assert result.container_result == container_result
        assert result.container_result.container_id == "test-id"

    def test_execute_code_with_docker_available(self, sandbox_manager):
        """Test code execution when Docker is available"""
        with patch.object(sandbox_manager.docker_executor, "is_available", return_value=True):
            with patch.object(sandbox_manager.risk_analyzer, "analyze_ast") as mock_risk:
                with patch.object(sandbox_manager.docker_executor, "execute_code") as mock_docker:
                    with patch.object(sandbox_manager.audit_logger, "log_execution") as mock_log:
                        # Mock risk analysis (allow execution)
                        mock_risk.return_value = RiskAssessment(
                            score=20, patterns=[], safe_to_execute=True, approval_required=False
                        )

                        # Mock Docker execution
                        mock_docker.return_value = {
                            "success": True,
                            "output": "Hello from Docker!",
                            "container_result": ContainerResult(
                                success=True,
                                container_id="test-container",
                                exit_code=0,
                                stdout="Hello from Docker!",
                            ),
                        }

                        # Execute request with Docker
                        request = ExecutionRequest(
                            code="print('Hello from Docker!')", use_docker=True
                        )

                        result = sandbox_manager.execute_code(request)

                        # Verify Docker was used
                        assert result.execution_method == "docker"
                        assert result.success is True
                        assert result.output == "Hello from Docker!"
                        assert result.container_result is not None

                        # Verify Docker executor was called
                        mock_docker.assert_called_once()

    def test_execute_code_fallback_to_local(self, sandbox_manager):
        """Test fallback to local execution when Docker unavailable"""
        with patch.object(sandbox_manager.docker_executor, "is_available", return_value=False):
            with patch.object(sandbox_manager.risk_analyzer, "analyze_ast") as mock_risk:
                with patch.object(sandbox_manager, "_execute_in_sandbox") as mock_local:
                    with patch.object(
                        sandbox_manager.resource_enforcer, "stop_monitoring"
                    ) as mock_monitoring:
                        # Mock risk analysis (allow execution)
                        mock_risk.return_value = RiskAssessment(
                            score=20, patterns=[], safe_to_execute=True, approval_required=False
                        )

                        # Mock local execution
                        mock_local.return_value = {"success": True, "output": "Hello from local!"}

                        # Mock resource monitoring
                        mock_monitoring.return_value = ResourceUsage(
                            cpu_percent=25.0,
                            memory_percent=30.0,
                            memory_used_gb=0.5,
                            elapsed_seconds=1.0,
                            approaching_limits=False,
                        )

                        # Execute request preferring Docker
                        request = ExecutionRequest(
                            code="print('Hello')",
                            use_docker=True,  # But Docker is unavailable
                        )

                        result = sandbox_manager.execute_code(request)

                        # Verify fallback to local execution
                        assert result.execution_method == "local"
                        assert result.success is True
                        assert result.output == "Hello from local!"
                        assert result.container_result is None

                        # Verify local execution was used
                        mock_local.assert_called_once()

    def test_execute_code_local_preference(self, sandbox_manager):
        """Test explicit preference for local execution"""
        with patch.object(sandbox_manager.risk_analyzer, "analyze_ast") as mock_risk:
            with patch.object(sandbox_manager, "_execute_in_sandbox") as mock_local:
                # Mock risk analysis (allow execution)
                mock_risk.return_value = RiskAssessment(
                    score=20, patterns=[], safe_to_execute=True, approval_required=False
                )

                # Mock local execution
                mock_local.return_value = {"success": True, "output": "Local execution"}

                # Execute request explicitly preferring local
                request = ExecutionRequest(
                    code="print('Local')",
                    use_docker=False,  # Explicitly prefer local
                )

                result = sandbox_manager.execute_code(request)

                # Verify local execution was used
                assert result.execution_method == "local"
                assert result.success is True

                # Docker executor should not be called
                sandbox_manager.docker_executor.execute_code.assert_not_called()

    def test_build_docker_config_from_request(self, sandbox_manager):
        """Test building Docker config from execution request"""
        from src.mai.sandbox.docker_executor import ContainerConfig

        # Use the actual method from DockerExecutor
        config = sandbox_manager.docker_executor._build_container_config(
            ContainerConfig(
                memory_limit="256m", cpu_limit="0.8", network_disabled=False, timeout_seconds=60
            ),
            {"TEST_VAR": "value"},
        )

        assert config["mem_limit"] == "256m"
        assert config["cpu_quota"] == 80000
        assert config["network_disabled"] is False
        assert config["security_opt"] is not None
        assert "TEST_VAR" in config["environment"]

    def test_get_docker_status(self, sandbox_manager, mock_docker_executor):
        """Test getting Docker status information"""
        sandbox_manager.docker_executor = mock_docker_executor

        status = sandbox_manager.get_docker_status()

        assert "available" in status
        assert "images" in status
        assert "system_info" in status
        assert status["available"] is True
        assert status["system_info"]["available"] is True

    def test_pull_docker_image(self, sandbox_manager, mock_docker_executor):
        """Test pulling Docker image"""
        sandbox_manager.docker_executor = mock_docker_executor
        mock_docker_executor.pull_image.return_value = True

        result = sandbox_manager.pull_docker_image("python:3.9-slim")

        assert result is True
        mock_docker_executor.pull_image.assert_called_once_with("python:3.9-slim")

    def test_cleanup_docker_containers(self, sandbox_manager, mock_docker_executor):
        """Test cleaning up Docker containers"""
        sandbox_manager.docker_executor = mock_docker_executor
        mock_docker_executor.cleanup_containers.return_value = 3

        result = sandbox_manager.cleanup_docker_containers()

        assert result == 3
        mock_docker_executor.cleanup_containers.assert_called_once()

    def test_get_system_status_includes_docker(self, sandbox_manager, mock_docker_executor):
        """Test system status includes Docker information"""
        sandbox_manager.docker_executor = mock_docker_executor

        with patch.object(sandbox_manager, "verify_log_integrity", return_value=True):
            status = sandbox_manager.get_system_status()

            assert "docker_available" in status
            assert "docker_info" in status
            assert status["docker_available"] is True
            assert status["docker_info"]["available"] is True

    def test_execute_code_with_additional_files(self, sandbox_manager):
        """Test code execution with additional files in Docker"""
        with patch.object(sandbox_manager.docker_executor, "is_available", return_value=True):
            with patch.object(sandbox_manager.risk_analyzer, "analyze_ast") as mock_risk:
                with patch.object(sandbox_manager.docker_executor, "execute_code") as mock_docker:
                    # Mock risk analysis (allow execution)
                    mock_risk.return_value = RiskAssessment(
                        score=20, patterns=[], safe_to_execute=True, approval_required=False
                    )

                    # Mock Docker execution
                    mock_docker.return_value = {
                        "success": True,
                        "output": "Processed files",
                        "container_result": ContainerResult(
                            success=True,
                            container_id="test-container",
                            exit_code=0,
                            stdout="Processed files",
                        ),
                    }

                    # Execute request with additional files
                    request = ExecutionRequest(
                        code="with open('data.txt', 'r') as f: print(f.read())",
                        use_docker=True,
                        additional_files={"data.txt": "test data content"},
                    )

                    result = sandbox_manager.execute_code(request)

                    # Verify Docker executor was called with files
                    mock_docker.assert_called_once()
                    call_args = mock_docker.call_args
                    assert "files" in call_args.kwargs
                    assert call_args.kwargs["files"] == {"data.txt": "test data content"}

                    assert result.success is True
                    assert result.execution_method == "docker"

    def test_risk_analysis_blocks_docker_execution(self, sandbox_manager):
        """Test that high-risk code is blocked even with Docker"""
        with patch.object(sandbox_manager.risk_analyzer, "analyze_ast") as mock_risk:
            # Mock high-risk analysis (block execution)
            mock_risk.return_value = RiskAssessment(
                score=85,
                patterns=[
                    RiskPattern(
                        pattern="os.system",
                        severity="BLOCKED",
                        score=50,
                        line_number=1,
                        description="System command execution",
                    )
                ],
                safe_to_execute=False,
                approval_required=True,
            )

            # Execute risky code with Docker preference
            request = ExecutionRequest(code="os.system('rm -rf /')", use_docker=True)

            result = sandbox_manager.execute_code(request)

            # Verify execution was blocked
            assert result.success is False
            assert "blocked" in result.error.lower()
            assert result.risk_assessment.score == 85
            assert result.execution_method == "local"  # Default before Docker check

            # Docker should not be called for blocked code
            sandbox_manager.docker_executor.execute_code.assert_not_called()


class TestSandboxManagerDockerEdgeCases:
    """Test edge cases and error handling in Docker integration"""

    @pytest.fixture
    def sandbox_manager(self):
        """Create SandboxManager instance for testing"""
        return SandboxManager()

    def test_docker_executor_error_handling(self, sandbox_manager):
        """Test handling of Docker executor errors"""
        with patch.object(sandbox_manager.docker_executor, "is_available", return_value=True):
            with patch.object(sandbox_manager.risk_analyzer, "analyze_ast") as mock_risk:
                with patch.object(sandbox_manager.docker_executor, "execute_code") as mock_docker:
                    # Mock risk analysis (allow execution)
                    mock_risk.return_value = RiskAssessment(
                        score=20, patterns=[], safe_to_execute=True, approval_required=False
                    )

                    # Mock Docker executor error
                    mock_docker.return_value = {
                        "success": False,
                        "error": "Docker daemon not available",
                        "container_result": None,
                    }

                    request = ExecutionRequest(code="print('test')", use_docker=True)

                    result = sandbox_manager.execute_code(request)

                    # Verify error handling
                    assert result.success is False
                    assert result.execution_method == "docker"
                    assert "Docker daemon not available" in result.error

    def test_container_resource_usage_integration(self, sandbox_manager):
        """Test integration of container resource usage"""
        with patch.object(sandbox_manager.docker_executor, "is_available", return_value=True):
            with patch.object(sandbox_manager.risk_analyzer, "analyze_ast") as mock_risk:
                with patch.object(sandbox_manager.docker_executor, "execute_code") as mock_docker:
                    # Mock risk analysis (allow execution)
                    mock_risk.return_value = RiskAssessment(
                        score=20, patterns=[], safe_to_execute=True, approval_required=False
                    )

                    # Mock Docker execution with resource usage
                    container_result = ContainerResult(
                        success=True,
                        container_id="test-container",
                        exit_code=0,
                        stdout="test output",
                        resource_usage={
                            "cpu_percent": 35.5,
                            "memory_usage_mb": 64.2,
                            "memory_percent": 12.5,
                        },
                    )

                    mock_docker.return_value = {
                        "success": True,
                        "output": "test output",
                        "container_result": container_result,
                    }

                    request = ExecutionRequest(code="print('test')", use_docker=True)

                    result = sandbox_manager.execute_code(request)

                    # Verify resource usage is preserved
                    assert result.container_result.resource_usage["cpu_percent"] == 35.5
                    assert result.container_result.resource_usage["memory_usage_mb"] == 64.2
                    assert result.container_result.resource_usage["memory_percent"] == 12.5


if __name__ == "__main__":
    pytest.main([__file__])

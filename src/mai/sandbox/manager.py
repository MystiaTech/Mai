"""
Sandbox Manager for Mai Safe Code Execution

Central orchestrator for sandbox execution, integrating risk analysis,
resource enforcement, and audit logging for safe code execution.
"""

import time
import uuid
from dataclasses import dataclass
from typing import Any

from .audit_logger import AuditLogger
from .docker_executor import ContainerConfig, ContainerResult, DockerExecutor
from .resource_enforcer import ResourceEnforcer, ResourceLimits, ResourceUsage
from .risk_analyzer import RiskAnalyzer, RiskAssessment


@dataclass
class ExecutionRequest:
    """Request for sandbox code execution"""

    code: str
    environment: dict[str, str] | None = None
    timeout_seconds: int = 30
    cpu_limit_percent: float = 70.0
    memory_limit_percent: float = 70.0
    network_allowed: bool = False
    filesystem_restricted: bool = True
    use_docker: bool = True
    docker_image: str = "python:3.10-slim"
    additional_files: dict[str, str] | None = None


@dataclass
class ExecutionResult:
    """Result of sandbox execution"""

    success: bool
    execution_id: str
    output: str | None = None
    error: str | None = None
    risk_assessment: RiskAssessment | None = None
    resource_usage: ResourceUsage | None = None
    execution_time: float = 0.0
    audit_entry_id: str | None = None
    execution_method: str = "local"  # "local", "docker", "fallback"
    container_result: ContainerResult | None = None


class SandboxManager:
    """
    Central sandbox orchestrator that coordinates risk analysis,
    resource enforcement, and audit logging for safe code execution.
    """

    def __init__(self, log_dir: str | None = None):
        """
        Initialize sandbox manager

        Args:
            log_dir: Directory for audit logs
        """
        self.risk_analyzer = RiskAnalyzer()
        self.resource_enforcer = ResourceEnforcer()
        self.audit_logger = AuditLogger(log_dir=log_dir)
        self.docker_executor = DockerExecutor(audit_logger=self.audit_logger)

        # Execution state
        self.active_executions: dict[str, dict[str, Any]] = {}

    def execute_code(self, request: ExecutionRequest) -> ExecutionResult:
        """
        Execute code in sandbox with full safety checks

        Args:
            request: ExecutionRequest with code and constraints

        Returns:
            ExecutionResult with execution details
        """
        execution_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        try:
            # Step 1: Risk analysis
            risk_assessment = self.risk_analyzer.analyze_ast(request.code)

            # Step 2: Check if execution is allowed
            if not self._is_execution_allowed(risk_assessment):
                result = ExecutionResult(
                    success=False,
                    execution_id=execution_id,
                    error=(
                        f"Code execution blocked: Risk score {risk_assessment.score} "
                        "exceeds safe threshold"
                    ),
                    risk_assessment=risk_assessment,
                    execution_time=time.time() - start_time,
                )

                # Log blocked execution
                self._log_execution(request, result, risk_assessment)
                return result

            # Step 3: Set resource limits
            resource_limits = ResourceLimits(
                cpu_percent=request.cpu_limit_percent,
                memory_percent=request.memory_limit_percent,
                timeout_seconds=request.timeout_seconds,
            )

            self.resource_enforcer.set_limits(resource_limits)
            self.resource_enforcer.start_monitoring()

            # Step 4: Choose execution method and execute code
            execution_method = (
                "docker" if request.use_docker and self.docker_executor.is_available() else "local"
            )

            if execution_method == "docker":
                execution_result = self._execute_in_docker(request, execution_id)
            else:
                execution_result = self._execute_in_sandbox(request, execution_id)
                execution_method = "local"

            # Step 5: Get resource usage (for local execution)
            if execution_method == "local":
                resource_usage = self.resource_enforcer.stop_monitoring()
            else:
                resource_usage = None  # Docker provides its own resource usage

            # Step 6: Create result
            result = ExecutionResult(
                success=execution_result.get("success", False),
                execution_id=execution_id,
                output=execution_result.get("output"),
                error=execution_result.get("error"),
                risk_assessment=risk_assessment,
                resource_usage=resource_usage,
                execution_time=time.time() - start_time,
                execution_method=execution_method,
                container_result=execution_result.get("container_result"),
            )

            # Step 7: Log execution
            audit_id = self._log_execution(request, result, risk_assessment, resource_usage)
            result.audit_entry_id = audit_id

            return result

        except Exception as e:
            # Handle unexpected errors
            result = ExecutionResult(
                success=False,
                execution_id=execution_id,
                error=f"Sandbox execution error: {str(e)}",
                execution_time=time.time() - start_time,
            )

            # Log error
            self._log_execution(request, result)
            return result

        finally:
            # Cleanup
            self.resource_enforcer.stop_monitoring()

    def check_risk(self, code: str) -> RiskAssessment:
        """
        Perform risk analysis on code

        Args:
            code: Code to analyze

        Returns:
            RiskAssessment with detailed analysis
        """
        return self.risk_analyzer.analyze_ast(code)

    def enforce_limits(self, limits: ResourceLimits) -> bool:
        """
        Set resource limits for execution

        Args:
            limits: Resource limits to enforce

        Returns:
            True if limits were set successfully
        """
        return self.resource_enforcer.set_limits(limits)

    def log_execution(
        self,
        code: str,
        execution_result: dict[str, Any],
        risk_assessment: dict[str, Any] | None = None,
        resource_usage: dict[str, Any] | None = None,
    ) -> str:
        """
        Log execution details to audit trail

        Args:
            code: Executed code
            execution_result: Result of execution
            risk_assessment: Risk analysis results
            resource_usage: Resource usage statistics

        Returns:
            Audit entry ID
        """
        return self.audit_logger.log_execution(
            code=code,
            execution_result=execution_result,
            risk_assessment=risk_assessment,
            resource_usage=resource_usage,
        )

    def get_execution_history(
        self, limit: int = 50, min_risk_score: int = 0
    ) -> list[dict[str, Any]]:
        """
        Get execution history from audit logs

        Args:
            limit: Maximum entries to return
            min_risk_score: Minimum risk score filter

        Returns:
            List of execution entries
        """
        return self.audit_logger.query_logs(limit=limit, risk_min=min_risk_score)

    def verify_log_integrity(self) -> bool:
        """
        Verify audit log integrity

        Returns:
            True if logs are intact
        """
        integrity = self.audit_logger.verify_integrity()
        return integrity.is_valid

    def get_system_status(self) -> dict[str, Any]:
        """
        Get current sandbox system status

        Returns:
            Dictionary with system status
        """
        return {
            "active_executions": len(self.active_executions),
            "resource_monitoring": self.resource_enforcer.monitoring_active,
            "current_usage": self.resource_enforcer.monitor_usage(),
            "log_stats": self.audit_logger.get_log_stats(),
            "log_integrity": self.verify_log_integrity(),
            "docker_available": self.docker_executor.is_available(),
            "docker_info": self.docker_executor.get_system_info(),
        }

    def get_docker_status(self) -> dict[str, Any]:
        """
        Get Docker executor status and available images

        Returns:
            Dictionary with Docker status
        """
        return {
            "available": self.docker_executor.is_available(),
            "images": self.docker_executor.get_available_images(),
            "system_info": self.docker_executor.get_system_info(),
        }

    def pull_docker_image(self, image_name: str) -> bool:
        """
        Pull a Docker image for execution

        Args:
            image_name: Name of the Docker image to pull

        Returns:
            True if image was pulled successfully
        """
        return self.docker_executor.pull_image(image_name)

    def cleanup_docker_containers(self) -> int:
        """
        Clean up any dangling Docker containers

        Returns:
            Number of containers cleaned up
        """
        return self.docker_executor.cleanup_containers()

    def _is_execution_allowed(self, risk_assessment: RiskAssessment) -> bool:
        """
        Determine if execution is allowed based on risk assessment

        Args:
            risk_assessment: Risk analysis result

        Returns:
            True if execution is allowed
        """
        # Block if any BLOCKED patterns detected
        blocked_patterns = [p for p in risk_assessment.patterns if p.severity == "BLOCKED"]
        if blocked_patterns:
            return False

        # Require approval for HIGH risk
        if risk_assessment.score >= 70:
            return False  # Would require user approval in full implementation

        return True

    def _execute_in_docker(self, request: ExecutionRequest, execution_id: str) -> dict[str, Any]:
        """
        Execute code in Docker container

        Args:
            request: Execution request
            execution_id: Unique execution identifier

        Returns:
            Dictionary with execution result
        """
        # Create container configuration based on request
        config = ContainerConfig(
            image=request.docker_image,
            timeout_seconds=request.timeout_seconds,
            memory_limit=f"{int(request.memory_limit_percent * 128 / 100)}m",  # Scale to container
            cpu_limit=str(request.cpu_limit_percent / 100),
            network_disabled=not request.network_allowed,
            read_only_filesystem=request.filesystem_restricted,
        )

        # Execute in Docker container
        container_result = self.docker_executor.execute_code(
            code=request.code,
            config=config,
            environment=request.environment,
            files=request.additional_files,
        )

        return {
            "success": container_result.success,
            "output": container_result.stdout,
            "error": container_result.stderr or container_result.error,
            "container_result": container_result,
        }

    def _execute_in_sandbox(self, request: ExecutionRequest, execution_id: str) -> dict[str, Any]:
        """
        Execute code in local sandbox environment (fallback)

        Args:
            request: Execution request
            execution_id: Unique execution identifier

        Returns:
            Dictionary with execution result
        """
        try:
            # For now, just simulate execution with eval (NOT PRODUCTION SAFE)
            # This would be replaced with proper sandbox execution
            if request.code.strip().startswith("print"):
                # Simple print statement
                result = eval(request.code)
                return {"success": True, "output": str(result)}
            else:
                # For safety, don't execute arbitrary code in this demo
                return {"success": False, "error": "Code execution not implemented in demo mode"}

        except Exception as e:
            return {"success": False, "error": f"Execution error: {str(e)}"}

    def _log_execution(
        self,
        request: ExecutionRequest,
        result: ExecutionResult,
        risk_assessment: RiskAssessment | None = None,
        resource_usage: ResourceUsage | None = None,
    ) -> str:
        """
        Internal method to log execution

        Args:
            request: Execution request
            result: Execution result
            risk_assessment: Risk analysis
            resource_usage: Resource usage

        Returns:
            Audit entry ID
        """
        # Prepare execution result for logging
        execution_result = {
            "success": result.success,
            "output": result.output,
            "error": result.error,
            "execution_time": result.execution_time,
        }

        # Prepare risk assessment for logging
        risk_data = None
        if risk_assessment:
            risk_data = {
                "score": risk_assessment.score,
                "patterns": [
                    {
                        "pattern": p.pattern,
                        "severity": p.severity,
                        "score": p.score,
                        "line_number": p.line_number,
                        "description": p.description,
                    }
                    for p in risk_assessment.patterns
                ],
                "safe_to_execute": risk_assessment.safe_to_execute,
                "approval_required": risk_assessment.approval_required,
            }

        # Prepare resource usage for logging
        usage_data = None
        if resource_usage:
            usage_data = {
                "cpu_percent": resource_usage.cpu_percent,
                "memory_percent": resource_usage.memory_percent,
                "memory_used_gb": resource_usage.memory_used_gb,
                "elapsed_seconds": resource_usage.elapsed_seconds,
                "approaching_limits": resource_usage.approaching_limits,
            }

        return self.audit_logger.log_execution(
            code=request.code,
            execution_result=execution_result,
            risk_assessment=risk_data,
            resource_usage=usage_data,
        )

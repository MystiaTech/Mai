"""Safety coordinator for orchestrating security assessment, sandbox execution, and audit logging."""

import logging
import psutil
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from ..security.assessor import SecurityAssessor, SecurityLevel
from ..sandbox.executor import SandboxExecutor
from ..audit.logger import AuditLogger

logger = logging.getLogger(__name__)


class SafetyCoordinator:
    """
    Main safety coordination logic that orchestrates all safety components.

    Coordinates security assessment, sandbox execution, and audit logging
    with user override capability and adaptive resource management.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize safety coordinator with component configurations.

        Args:
            config_path: Optional path to safety configuration directory
        """
        self.config_path = config_path or "config/safety"

        # Initialize components
        self.security_assessor = SecurityAssessor(
            config_path=str(Path(self.config_path) / "security.yaml")
        )
        self.sandbox_executor = SandboxExecutor(
            config_path=str(Path(self.config_path) / "sandbox.yaml")
        )
        self.audit_logger = AuditLogger(
            log_file=str(Path(self.config_path) / "audit.log"),
            storage_dir=str(Path(self.config_path) / "audit"),
        )

        # System resource monitoring
        self.system_monitor = psutil

    def execute_code_safely(
        self,
        code: str,
        user_override: bool = False,
        user_explanation: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Execute code with full safety coordination.

        Args:
            code: Python code to execute
            user_override: Whether user is overriding security decision
            user_explanation: Explanation for user override
            metadata: Additional execution metadata

        Returns:
            Execution result with comprehensive safety metadata
        """
        execution_id = self._generate_execution_id()
        metadata = metadata or {}
        metadata["execution_id"] = execution_id

        logger.info(f"Starting safe execution: {execution_id}")

        try:
            # Step 1: Security assessment
            security_level, security_findings = self.security_assessor.assess(code)

            # Log security assessment
            self.audit_logger.log_security_assessment(
                assessment={
                    "security_level": security_level.value,
                    "security_score": security_findings.get("security_score", 0),
                    "findings": security_findings,
                    "recommendations": security_findings.get("recommendations", []),
                },
                code_snippet=code,
                metadata=metadata,
            )

            # Step 2: Check if execution should proceed
            should_execute, override_used = self._check_execution_permission(
                security_level, user_override, user_explanation, metadata
            )

            if not should_execute:
                result = {
                    "success": False,
                    "blocked": True,
                    "security_level": security_level.value,
                    "security_findings": security_findings,
                    "reason": "Code blocked by security assessment",
                    "execution_id": execution_id,
                    "override_used": False,
                }

                # Log blocked execution
                self.audit_logger.log_security_event(
                    event_type="execution_blocked",
                    details={
                        "security_level": security_level.value,
                        "security_score": security_findings.get("security_score", 0),
                        "reason": result["reason"],
                    },
                    severity="HIGH",
                    metadata=metadata,
                )

                return result

            # Step 3: Determine adaptive resource limits
            trust_level = self._determine_trust_level(security_level, override_used)
            resource_limits = self._get_adaptive_resource_limits(code, trust_level)

            # Step 4: Execute in sandbox
            execution_result = self.sandbox_executor.execute_code(
                code, trust_level=trust_level
            )

            # Step 5: Log execution
            self.audit_logger.log_code_execution(
                code=code,
                result=execution_result,
                execution_time=execution_result.get("execution_time"),
                security_level=security_level.value,
                metadata={
                    **metadata,
                    "resource_limits": resource_limits,
                    "trust_level": trust_level,
                    "override_used": override_used,
                },
            )

            # Step 6: Return comprehensive result
            return {
                "success": execution_result["success"],
                "execution_id": execution_id,
                "security_level": security_level.value,
                "security_findings": security_findings,
                "execution_result": execution_result,
                "resource_limits": resource_limits,
                "trust_level": trust_level,
                "override_used": override_used,
                "blocked": False,
            }

        except Exception as e:
            logger.error(f"Safety coordination failed: {e}")

            # Log error
            self.audit_logger.log_security_event(
                event_type="safety_coordination_error",
                details={"error": str(e), "execution_id": execution_id},
                severity="CRITICAL",
                metadata=metadata,
            )

            return {
                "success": False,
                "execution_id": execution_id,
                "error": str(e),
                "blocked": False,
                "override_used": False,
            }

    def _check_execution_permission(
        self,
        security_level: SecurityLevel,
        user_override: bool,
        user_explanation: Optional[str],
        metadata: Dict,
    ) -> Tuple[bool, bool]:
        """
        Check if execution should proceed based on security level and user override.

        Args:
            security_level: Assessed security level
            user_override: Whether user requested override
            user_explanation: User's explanation for override
            metadata: Execution metadata

        Returns:
            Tuple of (should_execute, override_used)
        """
        override_used = False

        # LOW and MEDIUM: Always allow
        if security_level in [SecurityLevel.LOW, SecurityLevel.MEDIUM]:
            return True, override_used

        # HIGH: Allow with warning
        if security_level == SecurityLevel.HIGH:
            logger.warning(
                f"High risk code execution approved: {metadata.get('execution_id')}"
            )
            return True, override_used

        # BLOCKED: Only allow with explicit user override
        if security_level == SecurityLevel.BLOCKED:
            if user_override and user_explanation:
                override_used = True
                logger.warning(
                    f"User override for BLOCKED code: {user_explanation} "
                    f"(execution_id: {metadata.get('execution_id')})"
                )

                # Log override event
                self.audit_logger.log_security_event(
                    event_type="user_override",
                    details={
                        "original_security_level": security_level.value,
                        "user_explanation": user_explanation,
                    },
                    severity="HIGH",
                    metadata=metadata,
                )

                return True, override_used
            else:
                return False, override_used

        return False, override_used

    def _determine_trust_level(
        self, security_level: SecurityLevel, override_used: bool
    ) -> str:
        """
        Determine trust level for sandbox execution based on security assessment.

        Args:
            security_level: Assessed security level
            override_used: Whether user override was used

        Returns:
            Trust level string for sandbox configuration
        """
        if override_used:
            return "untrusted"  # Use most restrictive for overrides

        if security_level == SecurityLevel.LOW:
            return "trusted"
        elif security_level == SecurityLevel.MEDIUM:
            return "standard"
        else:  # HIGH or BLOCKED (with override)
            return "untrusted"

    def _get_adaptive_resource_limits(
        self, code: str, trust_level: str
    ) -> Dict[str, Any]:
        """
        Calculate adaptive resource limits based on code complexity and system resources.

        Args:
            code: Code to analyze for complexity
            trust_level: Trust level affecting base limits

        Returns:
            Resource limit configuration
        """
        # Get system resources
        cpu_count = self.system_monitor.cpu_count()
        memory_info = self.system_monitor.virtual_memory()
        available_memory_gb = memory_info.available / (1024**3)

        # Analyze code complexity
        code_lines = len(code.splitlines())
        code_complexity = self._analyze_code_complexity(code)

        # Base limits by trust level
        if trust_level == "trusted":
            base_cpu = min(cpu_count - 1, 4)
            base_memory = min(available_memory_gb * 0.3, 4.0)
            base_timeout = 300
        elif trust_level == "standard":
            base_cpu = min(cpu_count // 2, 2)
            base_memory = min(available_memory_gb * 0.15, 2.0)
            base_timeout = 120
        else:  # untrusted
            base_cpu = 1
            base_memory = min(available_memory_gb * 0.05, 0.5)
            base_timeout = 30

        # Adjust for code complexity
        complexity_multiplier = 1.0 + (code_complexity * 0.2)
        complexity_multiplier = min(complexity_multiplier, 2.0)  # Cap at 2x

        # Final limits
        resource_limits = {
            "cpu_count": max(1, int(base_cpu * complexity_multiplier)),
            "memory_limit_gb": round(base_memory * complexity_multiplier, 2),
            "timeout_seconds": int(base_timeout * complexity_multiplier),
            "code_lines": code_lines,
            "complexity_score": code_complexity,
            "system_cpu_count": cpu_count,
            "system_memory_gb": round(available_memory_gb, 2),
        }

        return resource_limits

    def _analyze_code_complexity(self, code: str) -> float:
        """
        Analyze code complexity for resource allocation.

        Args:
            code: Code to analyze

        Returns:
            Complexity score (0.0 to 1.0)
        """
        complexity_score = 0.0

        # Line count factor
        lines = len(code.splitlines())
        complexity_score += min(lines / 100, 0.3)  # Max 0.3 for line count

        # Control flow structures
        control_keywords = ["if", "for", "while", "try", "with", "def", "class"]
        for keyword in control_keywords:
            count = code.count(keyword)
            complexity_score += min(count / 20, 0.1)  # Max 0.1 per keyword type

        # Import statements
        import_count = code.count("import") + code.count("from")
        complexity_score += min(import_count / 10, 0.1)

        # String operations (potential for heavy processing)
        string_ops = code.count(".format") + code.count("f'") + code.count('f"')
        complexity_score += min(string_ops / 10, 0.1)

        return min(complexity_score, 1.0)

    def _generate_execution_id(self) -> str:
        """Generate unique execution ID."""
        import uuid

        return str(uuid.uuid4())[:8]

    def get_security_status(self) -> Dict[str, Any]:
        """
        Get current security system status.

        Returns:
            Security system status information
        """
        return {
            "security_assessor": "active",
            "sandbox_executor": "active",
            "audit_logger": "active",
            "system_resources": {
                "cpu_count": self.system_monitor.cpu_count(),
                "memory_total_gb": round(
                    self.system_monitor.virtual_memory().total / (1024**3), 2
                ),
                "memory_available_gb": round(
                    self.system_monitor.virtual_memory().available / (1024**3), 2
                ),
            },
            "audit_integrity": self.audit_logger.verify_integrity(),
        }

    def get_execution_history(self, limit: int = 10) -> Dict[str, Any]:
        """
        Get recent execution history from audit logs.

        Args:
            limit: Maximum number of entries to return

        Returns:
            Recent execution history
        """
        # Get security summary for recent executions
        summary = self.audit_logger.get_security_summary(time_range_hours=24)

        return {
            "summary": summary,
            "recent_executions": summary.get("code_executions", 0),
            "security_assessments": summary.get("security_assessments", 0),
            "resource_violations": summary.get("resource_violations", 0),
        }

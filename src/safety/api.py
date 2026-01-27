"""Public API interface for safety system."""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from .coordinator import SafetyCoordinator

logger = logging.getLogger(__name__)


class SafetyAPI:
    """
    Public interface for safety functionality.

    Provides clean, validated interface for other system components
    to use safety functionality including code assessment and execution.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize safety API with coordinator backend.

        Args:
            config_path: Optional path to safety configuration
        """
        self.coordinator = SafetyCoordinator(config_path)

    def assess_and_execute(
        self,
        code: str,
        user_override: bool = False,
        user_explanation: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Assess and execute code with full safety coordination.

        Args:
            code: Python code to assess and execute
            user_override: Whether user wants to override security decision
            user_explanation: Required explanation for override
            metadata: Additional execution metadata

        Returns:
            Formatted execution result with security metadata

        Raises:
            ValueError: If input validation fails
        """
        # Input validation
        validation_result = self._validate_code_input(
            code, user_override, user_explanation
        )
        if not validation_result["valid"]:
            raise ValueError(validation_result["error"])

        # Execute through coordinator
        result = self.coordinator.execute_code_safely(
            code=code,
            user_override=user_override,
            user_explanation=user_explanation,
            metadata=metadata,
        )

        # Format response
        return self._format_execution_response(result)

    def assess_code_only(self, code: str) -> Dict[str, Any]:
        """
        Assess code security without execution.

        Args:
            code: Python code to assess

        Returns:
            Security assessment results
        """
        if not code or not code.strip():
            raise ValueError("Code cannot be empty")

        security_level, findings = self.coordinator.security_assessor.assess(code)

        return {
            "security_level": security_level.value,
            "security_score": findings.get("security_score", 0),
            "findings": findings,
            "recommendations": findings.get("recommendations", []),
            "assessed_at": datetime.utcnow().isoformat(),
            "can_execute": security_level.value != "BLOCKED",
        }

    def get_execution_history(self, limit: int = 10) -> Dict[str, Any]:
        """
        Get recent execution history.

        Args:
            limit: Maximum number of entries to retrieve

        Returns:
            Formatted execution history
        """
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("Limit must be a positive integer")

        history = self.coordinator.get_execution_history(limit)

        return {
            "request": {"limit": limit},
            "response": history,
            "retrieved_at": datetime.utcnow().isoformat(),
        }

    def get_security_status(self) -> Dict[str, Any]:
        """
        Get current security system status.

        Returns:
            Security system status and health information
        """
        status = self.coordinator.get_security_status()

        return {
            "system_status": "operational"
            if all(
                component == "active"
                for component in [
                    status.get("security_assessor"),
                    status.get("sandbox_executor"),
                    status.get("audit_logger"),
                ]
            )
            else "degraded",
            "components": {
                "security_assessor": status.get("security_assessor"),
                "sandbox_executor": status.get("sandbox_executor"),
                "audit_logger": status.get("audit_logger"),
            },
            "system_resources": status.get("system_resources", {}),
            "audit_integrity": status.get("audit_integrity", {}),
            "status_checked_at": datetime.utcnow().isoformat(),
        }

    def configure_policies(self, policies: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update security and sandbox policies.

        Args:
            policies: Policy configuration dictionary

        Returns:
            Configuration update results
        """
        if not isinstance(policies, dict):
            raise ValueError("Policies must be a dictionary")

        update_results = {
            "updated_policies": [],
            "failed_updates": [],
            "validation_errors": [],
        }

        # Validate and update security policies
        if "security" in policies:
            try:
                self._validate_security_policies(policies["security"])
                # Note: In a real implementation, this would update the assessor config
                update_results["updated_policies"].append("security")
            except Exception as e:
                update_results["failed_updates"].append("security")
                update_results["validation_errors"].append(
                    f"Security policies: {str(e)}"
                )

        # Validate and update sandbox policies
        if "sandbox" in policies:
            try:
                self._validate_sandbox_policies(policies["sandbox"])
                # Note: In a real implementation, this would update the executor config
                update_results["updated_policies"].append("sandbox")
            except Exception as e:
                update_results["failed_updates"].append("sandbox")
                update_results["validation_errors"].append(
                    f"Sandbox policies: {str(e)}"
                )

        return {
            "request": {"policies": list(policies.keys())},
            "response": update_results,
            "updated_at": datetime.utcnow().isoformat(),
        }

    def get_audit_report(
        self, time_range_hours: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive audit report.

        Args:
            time_range_hours: Optional time filter for report

        Returns:
            Audit report data
        """
        if time_range_hours is not None:
            if not isinstance(time_range_hours, int) or time_range_hours <= 0:
                raise ValueError("time_range_hours must be a positive integer")

        # Get security summary
        summary = self.coordinator.audit_logger.get_security_summary(
            time_range_hours or 24
        )

        # Get integrity check
        integrity = self.coordinator.audit_logger.verify_integrity()

        return {
            "report_period_hours": time_range_hours or 24,
            "summary": summary,
            "integrity_check": integrity,
            "report_generated_at": datetime.utcnow().isoformat(),
        }

    def _validate_code_input(
        self, code: str, user_override: bool, user_explanation: Optional[str]
    ) -> Dict[str, Any]:
        """
        Validate code execution input parameters.

        Args:
            code: Code to validate
            user_override: Override flag
            user_explanation: Override explanation

        Returns:
            Validation result with error if invalid
        """
        if not code or not code.strip():
            return {"valid": False, "error": "Code cannot be empty"}

        if len(code) > 100000:  # 100KB limit
            return {"valid": False, "error": "Code too large (max 100KB)"}

        if user_override and not user_explanation:
            return {"valid": False, "error": "User override requires explanation"}

        if user_explanation and len(user_explanation) > 500:
            return {
                "valid": False,
                "error": "Override explanation too long (max 500 characters)",
            }

        return {"valid": True}

    def _format_execution_response(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format execution result for API response.

        Args:
            result: Raw execution result from coordinator

        Returns:
            Formatted API response
        """
        response = {
            "request_id": result.get("execution_id"),
            "success": result.get("success", False),
            "timestamp": datetime.utcnow().isoformat(),
            "security": {
                "level": result.get("security_level"),
                "override_used": result.get("override_used", False),
                "findings": result.get("security_findings", {}),
            },
        }

        if result.get("blocked"):
            response["blocked"] = True
            response["reason"] = result.get(
                "reason", "Security assessment blocked execution"
            )
        else:
            response["execution"] = result.get("execution_result", {})
            response["resource_limits"] = result.get("resource_limits", {})
            response["trust_level"] = result.get("trust_level")

        if "error" in result:
            response["error"] = result["error"]

        return response

    def _validate_security_policies(self, policies: Dict[str, Any]) -> None:
        """
        Validate security policy configuration.

        Args:
            policies: Security policies to validate

        Raises:
            ValueError: If policies are invalid
        """
        required_keys = ["blocked_patterns", "high_triggers", "thresholds"]
        for key in required_keys:
            if key not in policies:
                raise ValueError(f"Missing required security policy: {key}")

        # Validate thresholds
        thresholds = policies["thresholds"]
        if not all(isinstance(v, (int, float)) and v >= 0 for v in thresholds.values()):
            raise ValueError("Security thresholds must be non-negative numbers")

    def _validate_sandbox_policies(self, policies: Dict[str, Any]) -> None:
        """
        Validate sandbox policy configuration.

        Args:
            policies: Sandbox policies to validate

        Raises:
            ValueError: If policies are invalid
        """
        if "resources" in policies:
            resources = policies["resources"]

            # Validate timeout
            if "timeout" in resources and not (
                isinstance(resources["timeout"], (int, float))
                and resources["timeout"] > 0
            ):
                raise ValueError("Timeout must be a positive number")

            # Validate memory limit
            if "mem_limit" in resources:
                mem_limit = str(resources["mem_limit"])
                if not (mem_limit.endswith(("g", "m", "k")) or mem_limit.isdigit()):
                    raise ValueError("Memory limit must end with g/m/k or be a number")

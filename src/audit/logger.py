"""High-level audit logging interface for security events."""

import time
from datetime import datetime
from typing import Dict, Any, Optional, Union
from .crypto_logger import TamperProofLogger


class AuditLogger:
    """
    High-level interface for logging security events with tamper-proof protection.

    Provides convenient methods for logging different types of security events
    that are relevant to the Mai system.
    """

    def __init__(self, log_file: Optional[str] = None, storage_dir: str = "logs/audit"):
        """Initialize audit logger with tamper-proof backend."""
        self.crypto_logger = TamperProofLogger(log_file, storage_dir)

    def log_code_execution(
        self,
        code: str,
        result: Any,
        execution_time: Optional[float] = None,
        security_level: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Log code execution with comprehensive details.

        Args:
            code: Executed code
            result: Execution result
            execution_time: Time taken in seconds
            security_level: Security assessment level
            metadata: Additional execution metadata

        Returns:
            Hash of the logged entry
        """
        event_data = {
            "code": code,
            "code_length": len(code),
            "result_type": type(result).__name__,
            "result_summary": str(result)[:500]
            if result
            else None,  # Truncate long results
            "execution_time_seconds": execution_time,
            "security_level": security_level,
            "timestamp_utc": datetime.utcnow().isoformat(),
        }

        # Add resource usage if available
        if metadata and "resource_usage" in metadata:
            event_data["resource_usage"] = metadata["resource_usage"]

        log_metadata = {
            "category": "code_execution",
            "user": metadata.get("user") if metadata else None,
            "session": metadata.get("session") if metadata else None,
        }

        return self.crypto_logger.log_event("code_execution", event_data, log_metadata)

    def log_security_assessment(
        self,
        assessment: Dict[str, Any],
        code_snippet: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Log security assessment results.

        Args:
            assessment: Security assessment results from SecurityAssessor
            code_snippet: Assessed code snippet (truncated)
            metadata: Additional assessment metadata

        Returns:
            Hash of the logged entry
        """
        event_data = {
            "security_level": assessment.get("security_level"),
            "security_score": assessment.get("security_score"),
            "findings": assessment.get("findings", {}),
            "recommendations": assessment.get("recommendations", []),
            "assessment_timestamp": datetime.utcnow().isoformat(),
        }

        # Include code snippet if provided
        if code_snippet:
            event_data["code_snippet"] = code_snippet[:1000]  # Limit length

        # Extract key findings for quick reference
        findings = assessment.get("findings", {})
        event_data["summary"] = {
            "bandit_issues": len(findings.get("bandit_results", [])),
            "semgrep_issues": len(findings.get("semgrep_results", [])),
            "custom_issues": len(
                findings.get("custom_analysis", {}).get("blocked_patterns", [])
            ),
        }

        log_metadata = {
            "category": "security_assessment",
            "assessment_tool": "multi_tool_analysis",
            "user": metadata.get("user") if metadata else None,
            "session": metadata.get("session") if metadata else None,
        }

        return self.crypto_logger.log_event(
            "security_assessment", event_data, log_metadata
        )

    def log_container_creation(
        self,
        container_config: Dict[str, Any],
        container_id: Optional[str] = None,
        security_hardening: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Log container creation for code execution.

        Args:
            container_config: Container configuration
            container_id: Container ID/identifier
            security_hardening: Applied security measures
            metadata: Additional container metadata

        Returns:
            Hash of the logged entry
        """
        event_data = {
            "container_config": container_config,
            "container_id": container_id,
            "security_hardening": security_hardening or {},
            "creation_timestamp": datetime.utcnow().isoformat(),
        }

        # Extract security-relevant config
        security_config = {
            "cpu_limit": container_config.get("cpu_limit"),
            "memory_limit": container_config.get("memory_limit"),
            "network_mode": container_config.get("network_mode"),
            "read_only": container_config.get("read_only"),
            "user": container_config.get("user"),
            "capabilities_dropped": container_config.get("cap_drop"),
            "security_options": container_config.get("security_opt"),
        }
        event_data["security_config"] = security_config

        log_metadata = {
            "category": "container_creation",
            "orchestrator": "docker",
            "user": metadata.get("user") if metadata else None,
            "session": metadata.get("session") if metadata else None,
        }

        return self.crypto_logger.log_event(
            "container_creation", event_data, log_metadata
        )

    def log_resource_violation(
        self,
        violation: Dict[str, Any],
        container_id: Optional[str] = None,
        action_taken: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Log resource usage violations.

        Args:
            violation: Resource violation details
            container_id: Associated container ID
            action_taken: Action taken in response
            metadata: Additional violation metadata

        Returns:
            Hash of the logged entry
        """
        event_data = {
            "violation_type": violation.get("type"),
            "resource_type": violation.get("resource"),
            "threshold": violation.get("threshold"),
            "actual_value": violation.get("actual_value"),
            "container_id": container_id,
            "action_taken": action_taken,
            "violation_timestamp": datetime.utcnow().isoformat(),
        }

        # Add severity assessment
        severity = self._assess_violation_severity(violation)
        event_data["severity"] = severity

        log_metadata = {
            "category": "resource_violation",
            "monitoring_system": "docker_stats",
            "user": metadata.get("user") if metadata else None,
            "session": metadata.get("session") if metadata else None,
        }

        return self.crypto_logger.log_event(
            "resource_violation", event_data, log_metadata
        )

    def log_security_event(
        self,
        event_type: str,
        details: Dict[str, Any],
        severity: str = "INFO",
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Log general security events.

        Args:
            event_type: Type of security event
            details: Event details
            severity: Event severity (CRITICAL, HIGH, MEDIUM, LOW, INFO)
            metadata: Additional event metadata

        Returns:
            Hash of the logged entry
        """
        event_data = {
            "event_type": event_type,
            "severity": severity,
            "details": details,
            "event_timestamp": datetime.utcnow().isoformat(),
        }

        log_metadata = {
            "category": "security_event",
            "severity": severity,
            "user": metadata.get("user") if metadata else None,
            "session": metadata.get("session") if metadata else None,
        }

        return self.crypto_logger.log_event("security_event", event_data, log_metadata)

    def log_system_event(
        self, event_type: str, details: Dict[str, Any], metadata: Optional[Dict] = None
    ) -> str:
        """
        Log system-level events (startup, shutdown, configuration changes).

        Args:
            event_type: Type of system event
            details: Event details
            metadata: Additional event metadata

        Returns:
            Hash of the logged entry
        """
        event_data = {
            "system_event_type": event_type,
            "details": details,
            "event_timestamp": datetime.utcnow().isoformat(),
        }

        log_metadata = {
            "category": "system_event",
            "user": metadata.get("user") if metadata else None,
            "session": metadata.get("session") if metadata else None,
        }

        return self.crypto_logger.log_event("system_event", event_data, log_metadata)

    def _assess_violation_severity(self, violation: Dict[str, Any]) -> str:
        """
        Assess severity of resource violation.

        Args:
            violation: Violation details

        Returns:
            Severity level (CRITICAL, HIGH, MEDIUM, LOW)
        """
        violation_type = violation.get("type", "").lower()

        if violation_type in ["memory_oom", "cpu_exhaustion"]:
            return "CRITICAL"
        elif violation_type in ["memory_limit", "cpu_quota"]:
            return "HIGH"
        elif violation_type in ["disk_space", "network_io"]:
            return "MEDIUM"
        else:
            return "LOW"

    def get_security_summary(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """
        Get summary of security events in specified time range.

        Args:
            time_range_hours: Hours to look back

        Returns:
            Summary of security events
        """
        start_time = datetime.fromtimestamp(
            time.time() - (time_range_hours * 3600)
        ).isoformat()

        logs = self.crypto_logger.get_logs(start_time=start_time)

        summary = {
            "time_range_hours": time_range_hours,
            "total_events": len(logs),
            "event_types": {},
            "security_levels": {},
            "resource_violations": 0,
            "code_executions": 0,
            "security_assessments": 0,
        }

        for log in logs:
            event_type = log.get("event_type")

            # Count event types
            summary["event_types"][event_type] = (
                summary["event_types"].get(event_type, 0) + 1
            )

            # Count specific categories
            if event_type == "code_execution":
                summary["code_executions"] += 1
            elif event_type == "security_assessment":
                summary["security_assessments"] += 1
            elif event_type == "resource_violation":
                summary["resource_violations"] += 1

            # Count security levels for assessments
            if event_type == "security_assessment":
                level = log.get("event_data", {}).get("security_level", "UNKNOWN")
                summary["security_levels"][level] = (
                    summary["security_levels"].get(level, 0) + 1
                )

        return summary

    def verify_integrity(self) -> Dict[str, Any]:
        """
        Verify the integrity of the audit log chain.

        Returns:
            Integrity verification results
        """
        return self.crypto_logger.verify_chain()

    def export_audit_report(
        self, output_file: str, time_range_hours: Optional[int] = None
    ) -> bool:
        """
        Export comprehensive audit report.

        Args:
            output_file: Output file path
            time_range_hours: Optional time filter

        Returns:
            True if export successful
        """
        # Get filtered logs if time range specified
        if time_range_hours:
            start_time = datetime.fromtimestamp(
                time.time() - (time_range_hours * 3600)
            ).isoformat()
            logs = self.crypto_logger.get_logs(start_time=start_time)
        else:
            logs = self.crypto_logger.get_logs()

        # Create comprehensive report
        report = {
            "audit_report": {
                "generated_at": datetime.utcnow().isoformat(),
                "time_range_hours": time_range_hours,
                "total_entries": len(logs),
                "integrity_check": self.verify_integrity(),
                "security_summary": self.get_security_summary(time_range_hours or 24),
            },
            "logs": logs,
        }

        try:
            import json

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            return True
        except (IOError, json.JSONEncodeError):
            return False

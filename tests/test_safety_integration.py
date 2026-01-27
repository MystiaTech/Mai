"""Integration tests for safety system."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import tempfile
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.safety.coordinator import SafetyCoordinator
from src.safety.api import SafetyAPI
from src.security.assessor import SecurityLevel


class TestSafetyIntegration:
    """Integration tests for complete safety system."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary config directory for tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            yield config_dir

    @pytest.fixture
    def mock_coordinator(self):
        """Create safety coordinator with mocked components."""
        with patch("src.safety.coordinator.SecurityAssessor") as mock_assessor, patch(
            "src.safety.coordinator.SandboxExecutor"
        ) as mock_executor, patch(
            "src.safety.coordinator.AuditLogger"
        ) as mock_logger, patch("src.safety.coordinator.psutil") as mock_psutil:
            # Configure mock assessor
            assessor_instance = Mock()
            mock_assessor.return_value = assessor_instance

            # Configure mock executor
            executor_instance = Mock()
            mock_executor.return_value = executor_instance

            # Configure mock logger
            logger_instance = Mock()
            logger_instance.log_security_assessment.return_value = "hash1"
            logger_instance.log_code_execution.return_value = "hash2"
            logger_instance.log_security_event.return_value = "hash3"
            mock_logger.return_value = logger_instance

            # Configure mock psutil
            mock_psutil.cpu_count.return_value = 4
            mock_memory = Mock()
            mock_memory.available = 8 * 1024**3  # 8GB
            mock_psutil.virtual_memory.return_value = mock_memory

            coordinator = SafetyCoordinator()
            coordinator.security_assessor = assessor_instance
            coordinator.sandbox_executor = executor_instance
            coordinator.audit_logger = logger_instance

            yield coordinator, assessor_instance, executor_instance, logger_instance

    @pytest.fixture
    def safety_api(self, mock_coordinator):
        """Create safety API with mocked coordinator."""
        coordinator, _, _, _ = mock_coordinator

        # Mock the coordinator constructor
        with patch("src.safety.api.SafetyCoordinator", return_value=coordinator):
            api = SafetyAPI()
            api.coordinator = coordinator
            yield api

    def test_low_risk_code_executes_successfully(self, mock_coordinator):
        """Test that LOW risk code executes successfully."""
        coordinator, mock_assessor, mock_executor, _ = mock_coordinator

        # Setup LOW risk assessment
        mock_assessor.assess.return_value = (
            SecurityLevel.LOW,
            {"security_score": 1, "recommendations": ["Code appears safe"]},
        )

        # Setup successful execution
        mock_executor.execute_code.return_value = {
            "success": True,
            "exit_code": 0,
            "stdout": "Hello, World!",
            "stderr": "",
            "execution_time": 0.5,
            "resource_usage": {"cpu": 10, "memory": "50MB"},
        }

        # Execute code
        result = coordinator.execute_code_safely("print('Hello, World!')")

        # Verify result
        assert result["success"] is True
        assert result["security_level"] == "LOW"
        assert result["blocked"] is False
        assert result["override_used"] is False
        assert result["execution_result"]["stdout"] == "Hello, World!"

        # Verify components called correctly
        mock_assessor.assess.assert_called_once_with("print('Hello, World!')")
        mock_executor.execute_code.assert_called_once()

    def test_medium_risk_executes_with_warnings(self, mock_coordinator):
        """Test that MEDIUM risk code executes with warnings."""
        coordinator, mock_assessor, mock_executor, _ = mock_coordinator

        # Setup MEDIUM risk assessment
        mock_assessor.assess.return_value = (
            SecurityLevel.MEDIUM,
            {"security_score": 5, "recommendations": ["Review file operations"]},
        )

        # Setup successful execution
        mock_executor.execute_code.return_value = {
            "success": True,
            "exit_code": 0,
            "stdout": "File processed",
            "stderr": "",
            "execution_time": 1.2,
            "resource_usage": {"cpu": 25, "memory": "100MB"},
        }

        # Execute code
        result = coordinator.execute_code_safely("open('file.txt').read()")

        # Verify result
        assert result["success"] is True
        assert result["security_level"] == "MEDIUM"
        assert result["trust_level"] == "standard"
        assert result["blocked"] is False

        # Verify proper trust level was determined
        mock_executor.execute_code.assert_called_once_with(
            "open('file.txt').read()", trust_level="standard"
        )

    def test_high_risk_requires_user_confirmation(self, mock_coordinator):
        """Test that HIGH risk code can execute with user aware."""
        coordinator, mock_assessor, mock_executor, _ = mock_coordinator

        # Setup HIGH risk assessment
        mock_assessor.assess.return_value = (
            SecurityLevel.HIGH,
            {"security_score": 8, "recommendations": ["Avoid system calls"]},
        )

        # Setup successful execution
        mock_executor.execute_code.return_value = {
            "success": True,
            "exit_code": 0,
            "stdout": "System info retrieved",
            "stderr": "",
            "execution_time": 2.0,
            "resource_usage": {"cpu": 40, "memory": "200MB"},
        }

        # Execute code
        result = coordinator.execute_code_safely("import os; os.system('uname')")

        # Verify result
        assert result["success"] is True
        assert result["security_level"] == "HIGH"
        assert result["trust_level"] == "untrusted"
        assert result["blocked"] is False

        # Verify untrusted trust level for HIGH risk
        mock_executor.execute_code.assert_called_once_with(
            "import os; os.system('uname')", trust_level="untrusted"
        )

    def test_blocked_code_blocked_without_override(self, mock_coordinator):
        """Test that BLOCKED code is blocked without override."""
        coordinator, mock_assessor, mock_executor, _ = mock_coordinator

        # Setup BLOCKED assessment
        mock_assessor.assess.return_value = (
            SecurityLevel.BLOCKED,
            {"security_score": 15, "recommendations": ["Remove dangerous functions"]},
        )

        # Execute code without override
        result = coordinator.execute_code_safely(
            'eval(\'__import__("os").system("rm -rf /")\')'
        )

        # Verify result
        assert result["success"] is False
        assert result["blocked"] is True
        assert result["security_level"] == "BLOCKED"
        assert result["override_used"] is False
        assert "blocked by security assessment" in result["reason"]

        # Verify executor was not called
        mock_executor.execute_code.assert_not_called()

        # Verify security event logged
        coordinator.audit_logger.log_security_event.assert_called()
        call_args = coordinator.audit_logger.log_security_event.call_args
        assert call_args[1]["event_type"] == "execution_blocked"

    def test_blocked_code_executes_with_user_override(self, mock_coordinator):
        """Test that BLOCKED code executes with user override."""
        coordinator, mock_assessor, mock_executor, _ = mock_coordinator

        # Setup BLOCKED assessment
        mock_assessor.assess.return_value = (
            SecurityLevel.BLOCKED,
            {"security_score": 15, "recommendations": ["Remove dangerous functions"]},
        )

        # Setup successful execution for override case
        mock_executor.execute_code.return_value = {
            "success": True,
            "exit_code": 0,
            "stdout": "Command executed",
            "stderr": "",
            "execution_time": 1.5,
            "resource_usage": {"cpu": 30, "memory": "150MB"},
        }

        # Execute code with override
        result = coordinator.execute_code_safely(
            'eval(\'__import__("os").system("echo test")\')',
            user_override=True,
            user_explanation="Testing eval functionality for educational purposes",
        )

        # Verify result
        assert result["success"] is True
        assert result["security_level"] == "BLOCKED"
        assert result["blocked"] is False
        assert result["override_used"] is True
        assert result["trust_level"] == "untrusted"

        # Verify executor was called with untrusted level
        mock_executor.execute_code.assert_called_once_with(
            'eval(\'__import__("os").system("echo test")\')', trust_level="untrusted"
        )

        # Verify override logged
        coordinator.audit_logger.log_security_event.assert_called()
        call_args = coordinator.audit_logger.log_security_event.call_args
        assert call_args[1]["event_type"] == "user_override"

    def test_resource_limits_adapt_to_code_complexity(self, mock_coordinator):
        """Test that resource limits adapt based on code complexity."""
        coordinator, mock_assessor, mock_executor, _ = mock_coordinator

        # Setup LOW risk assessment
        mock_assessor.assess.return_value = (
            SecurityLevel.LOW,
            {"security_score": 1, "recommendations": []},
        )

        # Setup successful execution
        mock_executor.execute_code.return_value = {
            "success": True,
            "exit_code": 0,
            "stdout": "success",
            "stderr": "",
            "execution_time": 0.5,
            "resource_usage": {"cpu": 10, "memory": "50MB"},
        }

        # Simple code should get lower limits
        simple_code = "print('hello')"
        result = coordinator.execute_code_safely(simple_code)
        simple_limits = result["resource_limits"]

        # Complex code should get higher limits
        complex_code = """
def complex_function(data):
    result = []
    for item in data:
        if item % 2 == 0:
            result.append(item * 2)
        else:
            result.append(item + 1)
    return result

# Test the function
test_data = list(range(100))
output = complex_function(test_data)
print(f"Processed {len(output)} items")
"""
        result = coordinator.execute_code_safely(complex_code)
        complex_limits = result["resource_limits"]

        # Complex code should have higher complexity score
        assert complex_limits["complexity_score"] > simple_limits["complexity_score"]
        assert complex_limits["code_lines"] > simple_limits["code_lines"]

    def test_audit_logs_created_for_all_operations(self, mock_coordinator):
        """Test that audit logs are created for all safety operations."""
        coordinator, mock_assessor, mock_executor, mock_logger = mock_coordinator

        # Setup assessment
        mock_assessor.assess.return_value = (
            SecurityLevel.MEDIUM,
            {"security_score": 5, "recommendations": []},
        )

        # Setup execution
        mock_executor.execute_code.return_value = {
            "success": True,
            "exit_code": 0,
            "stdout": "success",
            "stderr": "",
            "execution_time": 0.5,
            "resource_usage": {},
        }

        # Execute code
        coordinator.execute_code_safely("print('test')")

        # Verify all audit methods were called
        mock_logger.log_security_assessment.assert_called_once()
        mock_logger.log_code_execution.assert_called_once()

        # Check log call arguments
        assessment_call = mock_logger.log_security_assessment.call_args[1]
        assert "assessment" in assessment_call
        assert "code_snippet" in assessment_call

        execution_call = mock_logger.log_code_execution.call_args[1]
        assert "code" in execution_call
        assert "result" in execution_call
        assert "security_level" in execution_call

    @patch("src.audit.crypto_logger.TamperProofLogger")
    def test_hash_chain_tampering_detected(
        self, mock_crypto_logger_class, temp_config_dir
    ):
        """Test that hash chain tampering is detected."""
        # Setup mock logger to simulate tampering
        mock_crypto_logger = Mock()
        mock_crypto_logger.verify_chain.return_value = {
            "valid": False,
            "tampered_entry": {"hash": "invalid_hash"},
            "error": "Hash chain integrity compromised",
        }
        mock_crypto_logger_class.return_value = mock_crypto_logger

        # Create coordinator with tampered logger
        coordinator = SafetyCoordinator(str(temp_config_dir))
        coordinator.audit_logger.crypto_logger = mock_crypto_logger

        # Get security status
        status = coordinator.get_security_status()

        # Verify tampering is detected
        assert status["audit_integrity"]["valid"] is False
        assert "tampered_entry" in status["audit_integrity"]

    def test_api_assess_and_execute_interface(self, safety_api):
        """Test API assess_and_execute interface."""
        api = safety_api
        coordinator = api.coordinator

        # Mock coordinator response
        coordinator.execute_code_safely = Mock(
            return_value={
                "success": True,
                "execution_id": "test123",
                "security_level": "LOW",
                "security_findings": {"security_score": 1},
                "execution_result": {"stdout": "success"},
                "resource_limits": {"cpu_count": 2},
                "trust_level": "trusted",
                "override_used": False,
                "blocked": False,
            }
        )

        # Test API call
        result = api.assess_and_execute("print('test')")

        # Verify formatted response
        assert "request_id" in result
        assert result["success"] is True
        assert result["security"]["level"] == "LOW"
        assert result["execution"]["stdout"] == "success"
        assert "timestamp" in result

    def test_api_input_validation(self, safety_api):
        """Test API input validation."""
        api = safety_api

        # Test empty code
        with pytest.raises(ValueError, match="Code cannot be empty"):
            api.assess_and_execute("")

        # Test override without explanation
        with pytest.raises(ValueError, match="User override requires explanation"):
            api.assess_and_execute("print('test')", user_override=True)

        # Test code size limit
        large_code = "x" * 100001  # Over 100KB
        with pytest.raises(ValueError, match="Code too large"):
            api.assess_and_execute(large_code)

    def test_api_get_execution_history(self, safety_api):
        """Test API execution history retrieval."""
        api = safety_api
        coordinator = api.coordinator

        # Mock coordinator response
        coordinator.get_execution_history = Mock(
            return_value={
                "summary": {"code_executions": 5},
                "recent_executions": 5,
                "security_assessments": 10,
                "resource_violations": 0,
            }
        )

        # Test API call
        result = api.get_execution_history(limit=10)

        # Verify formatted response
        assert result["request"]["limit"] == 10
        assert result["response"]["recent_executions"] == 5
        assert "retrieved_at" in result

    def test_api_get_security_status(self, safety_api):
        """Test API security status retrieval."""
        api = safety_api
        coordinator = api.coordinator

        # Mock coordinator response
        coordinator.get_security_status = Mock(
            return_value={
                "security_assessor": "active",
                "sandbox_executor": "active",
                "audit_logger": "active",
                "system_resources": {"cpu_count": 4},
                "audit_integrity": {"valid": True},
            }
        )

        # Test API call
        result = api.get_security_status()

        # Verify formatted response
        assert result["system_status"] == "operational"
        assert result["components"]["security_assessor"] == "active"
        assert "status_checked_at" in result

    def test_api_configure_policies_validation(self, safety_api):
        """Test API policy configuration validation."""
        api = safety_api

        # Test invalid policy format
        with pytest.raises(ValueError, match="Policies must be a dictionary"):
            api.configure_policies("invalid")

        # Test invalid security policies
        invalid_security = {
            "security": {
                "blocked_patterns": [],
                "thresholds": {"blocked_score": "invalid"},  # Should be number
            }
        }
        result = api.configure_policies(invalid_security)
        assert "validation_errors" in result["response"]
        assert "security" in result["response"]["failed_updates"]

        # Test valid policies
        valid_policies = {
            "security": {
                "blocked_patterns": ["os.system"],
                "high_triggers": ["admin"],
                "thresholds": {"blocked_score": 10, "high_score": 7, "medium_score": 4},
            }
        }
        result = api.configure_policies(valid_policies)
        assert "security" in result["response"]["updated_policies"]

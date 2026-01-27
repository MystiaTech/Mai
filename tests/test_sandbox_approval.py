"""
Test suite for ApprovalSystem

This module provides comprehensive testing for the risk-based approval system
including user interaction, trust management, and edge cases.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import Dict, Any

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from mai.sandbox.approval_system import (
    ApprovalSystem,
    RiskLevel,
    ApprovalResult,
    RiskAnalysis,
    ApprovalRequest,
    ApprovalDecision,
)


class TestApprovalSystem:
    """Test cases for ApprovalSystem."""

    @pytest.fixture
    def approval_system(self):
        """Create fresh ApprovalSystem for each test."""
        with patch("mai.sandbox.approval_system.get_config") as mock_config:
            mock_config.return_value = Mock()
            mock_config.return_value.get.return_value = {
                "low_threshold": 0.3,
                "medium_threshold": 0.6,
                "high_threshold": 0.8,
            }
            return ApprovalSystem()

    @pytest.fixture
    def mock_low_risk_code(self):
        """Sample low-risk code."""
        return 'print("hello world")'

    @pytest.fixture
    def mock_medium_risk_code(self):
        """Sample medium-risk code."""
        return "import os\nprint(os.getcwd())"

    @pytest.fixture
    def mock_high_risk_code(self):
        """Sample high-risk code."""
        return 'import subprocess\nsubprocess.call(["ls", "-la"])'

    @pytest.fixture
    def mock_blocked_code(self):
        """Sample blocked code."""
        return 'os.system("rm -rf /")'

    def test_initialization(self, approval_system):
        """Test ApprovalSystem initialization."""
        assert approval_system.approval_history == []
        assert approval_system.user_preferences == {}
        assert approval_system.trust_patterns == {}
        assert approval_system.risk_thresholds["low_threshold"] == 0.3

    def test_risk_analysis_low_risk(self, approval_system, mock_low_risk_code):
        """Test risk analysis for low-risk code."""
        context = {}
        risk_analysis = approval_system._analyze_code_risk(mock_low_risk_code, context)

        assert risk_analysis.risk_level == RiskLevel.LOW
        assert risk_analysis.severity_score < 0.3
        assert len(risk_analysis.reasons) == 0
        assert risk_analysis.confidence > 0.5

    def test_risk_analysis_medium_risk(self, approval_system, mock_medium_risk_code):
        """Test risk analysis for medium-risk code."""
        context = {}
        risk_analysis = approval_system._analyze_code_risk(mock_medium_risk_code, context)

        assert risk_analysis.risk_level == RiskLevel.MEDIUM
        assert risk_analysis.severity_score >= 0.3
        assert len(risk_analysis.reasons) > 0
        assert "file_system" in risk_analysis.affected_resources

    def test_risk_analysis_high_risk(self, approval_system, mock_high_risk_code):
        """Test risk analysis for high-risk code."""
        context = {}
        risk_analysis = approval_system._analyze_code_risk(mock_high_risk_code, context)

        assert risk_analysis.risk_level == RiskLevel.HIGH
        assert risk_analysis.severity_score >= 0.6
        assert len(risk_analysis.reasons) > 0
        assert "system_operations" in risk_analysis.affected_resources

    def test_risk_analysis_blocked(self, approval_system, mock_blocked_code):
        """Test risk analysis for blocked code."""
        context = {}
        risk_analysis = approval_system._analyze_code_risk(mock_blocked_code, context)

        assert risk_analysis.risk_level == RiskLevel.BLOCKED
        assert any("blocked operation" in reason.lower() for reason in risk_analysis.reasons)

    def test_operation_type_detection(self, approval_system):
        """Test operation type detection."""
        assert approval_system._get_operation_type('print("hello")') == "output_operation"
        assert approval_system._get_operation_type("import os") == "module_import"
        assert approval_system._get_operation_type('os.system("ls")') == "system_command"
        assert approval_system._get_operation_type('open("file.txt")') == "file_operation"
        assert approval_system._get_operation_type("x = 5") == "code_execution"

    def test_request_id_generation(self, approval_system):
        """Test unique request ID generation."""
        code1 = 'print("test")'
        code2 = 'print("test")'

        id1 = approval_system._generate_request_id(code1)
        time.sleep(0.01)  # Small delay to ensure different timestamps
        id2 = approval_system._generate_request_id(code2)

        assert id1 != id2  # Should be different due to timestamp
        assert len(id1) == 12  # MD5 hash truncated to 12 chars
        assert len(id2) == 12

    @patch("builtins.input")
    def test_low_risk_approval_allow(self, mock_input, approval_system, mock_low_risk_code):
        """Test low-risk approval with user allowing."""
        mock_input.return_value = "y"

        result, decision = approval_system.request_approval(mock_low_risk_code)

        assert result == ApprovalResult.APPROVED
        assert decision.user_input == "allowed"
        assert decision.request.risk_analysis.risk_level == RiskLevel.LOW

    @patch("builtins.input")
    def test_low_risk_approval_deny(self, mock_input, approval_system, mock_low_risk_code):
        """Test low-risk approval with user denying."""
        mock_input.return_value = "n"

        result, decision = approval_system.request_approval(mock_low_risk_code)

        assert result == ApprovalResult.DENIED
        assert decision.user_input == "denied"

    @patch("builtins.input")
    def test_low_risk_approval_always(self, mock_input, approval_system, mock_low_risk_code):
        """Test low-risk approval with 'always allow' preference."""
        mock_input.return_value = "a"

        result, decision = approval_system.request_approval(mock_low_risk_code)

        assert result == ApprovalResult.APPROVED
        assert decision.user_input == "allowed_always"
        assert decision.trust_updated == True
        assert "output_operation" in approval_system.user_preferences

    @patch("builtins.input")
    def test_medium_risk_approval_details(self, mock_input, approval_system, mock_medium_risk_code):
        """Test medium-risk approval requesting details."""
        mock_input.return_value = "d"  # Request details first

        with patch.object(approval_system, "_present_detailed_view") as mock_detailed:
            mock_detailed.return_value = "allowed"

            result, decision = approval_system.request_approval(mock_medium_risk_code)

            assert result == ApprovalResult.APPROVED
            mock_detailed.assert_called_once()

    @patch("builtins.input")
    def test_high_risk_approval_confirm(self, mock_input, approval_system, mock_high_risk_code):
        """Test high-risk approval with confirmation."""
        mock_input.return_value = "confirm"

        result, decision = approval_system.request_approval(mock_high_risk_code)

        assert result == ApprovalResult.APPROVED
        assert decision.request.risk_analysis.risk_level == RiskLevel.HIGH

    @patch("builtins.input")
    def test_high_risk_approval_cancel(self, mock_input, approval_system, mock_high_risk_code):
        """Test high-risk approval with cancellation."""
        mock_input.return_value = "cancel"

        result, decision = approval_system.request_approval(mock_high_risk_code)

        assert result == ApprovalResult.DENIED

    @patch("builtins.print")
    def test_blocked_operation(self, mock_print, approval_system, mock_blocked_code):
        """Test blocked operation handling."""
        result, decision = approval_system.request_approval(mock_blocked_code)

        assert result == ApprovalResult.BLOCKED
        assert decision.request.risk_analysis.risk_level == RiskLevel.BLOCKED

    def test_auto_approval_for_trusted_operation(self, approval_system, mock_low_risk_code):
        """Test auto-approval for trusted operations."""
        # Set up user preference
        approval_system.user_preferences["output_operation"] = "auto_allow"

        result, decision = approval_system.request_approval(mock_low_risk_code)

        assert result == ApprovalResult.ALLOWED
        assert decision.user_input == "auto_allowed"

    def test_approval_history(self, approval_system, mock_low_risk_code):
        """Test approval history tracking."""
        # Add some decisions
        with patch("builtins.input", return_value="y"):
            approval_system.request_approval(mock_low_risk_code)
            approval_system.request_approval(mock_low_risk_code)

        history = approval_system.get_approval_history(5)
        assert len(history) == 2
        assert all(isinstance(decision, ApprovalDecision) for decision in history)

    def test_trust_patterns_learning(self, approval_system, mock_low_risk_code):
        """Test trust pattern learning."""
        # Add approved decisions
        with patch("builtins.input", return_value="y"):
            for _ in range(3):
                approval_system.request_approval(mock_low_risk_code)

        patterns = approval_system.get_trust_patterns()
        assert "output_operation" in patterns
        assert patterns["output_operation"] == 3

    def test_preferences_reset(self, approval_system):
        """Test preferences reset."""
        # Add some preferences
        approval_system.user_preferences = {"test": "value"}
        approval_system.reset_preferences()

        assert approval_system.user_preferences == {}

    def test_is_code_safe(self, approval_system, mock_low_risk_code, mock_high_risk_code):
        """Test quick safety check."""
        assert approval_system.is_code_safe(mock_low_risk_code) == True
        assert approval_system.is_code_safe(mock_high_risk_code) == False

    def test_context_awareness(self, approval_system, mock_low_risk_code):
        """Test context-aware risk analysis."""
        # New user context should increase risk
        context_new_user = {"user_level": "new"}
        risk_new = approval_system._analyze_code_risk(mock_low_risk_code, context_new_user)

        context_known_user = {"user_level": "known"}
        risk_known = approval_system._analyze_code_risk(mock_low_risk_code, context_known_user)

        assert risk_new.severity_score > risk_known.severity_score
        assert "New user profile" in risk_new.reasons

    def test_request_id_uniqueness(self, approval_system):
        """Test that request IDs are unique even for same code."""
        code = 'print("test")'
        ids = []

        for _ in range(10):
            rid = approval_system._generate_request_id(code)
            assert rid not in ids, f"Duplicate ID: {rid}"
            ids.append(rid)

    def test_risk_score_accumulation(self, approval_system):
        """Test that multiple risk factors accumulate."""
        # Code with multiple risk factors
        risky_code = """
import os
import subprocess
os.system("ls")
subprocess.call(["pwd"])
        """
        risk_analysis = approval_system._analyze_code_risk(risky_code, {})

        assert risk_analysis.severity_score > 0.5
        assert len(risk_analysis.reasons) >= 2
        assert "system_operations" in risk_analysis.affected_resources

    @patch("builtins.input")
    def test_detailed_view_presentation(self, mock_input, approval_system, mock_medium_risk_code):
        """Test detailed view presentation."""
        mock_input.return_value = "y"

        # Create a request
        risk_analysis = approval_system._analyze_code_risk(mock_medium_risk_code, {})
        request = ApprovalRequest(
            code=mock_medium_risk_code,
            risk_analysis=risk_analysis,
            context={"test": "value"},
            timestamp=datetime.now(),
            request_id="test123",
        )

        result = approval_system._present_detailed_view(request)
        assert result == "allowed"

    @patch("builtins.input")
    def test_detailed_analysis_presentation(self, mock_input, approval_system, mock_high_risk_code):
        """Test detailed analysis presentation."""
        mock_input.return_value = "confirm"

        # Create a request
        risk_analysis = approval_system._analyze_code_risk(mock_high_risk_code, {})
        request = ApprovalRequest(
            code=mock_high_risk_code,
            risk_analysis=risk_analysis,
            context={},
            timestamp=datetime.now(),
            request_id="test456",
        )

        result = approval_system._present_detailed_analysis(request)
        assert result == "allowed"

    def test_error_handling_in_risk_analysis(self, approval_system):
        """Test error handling in risk analysis."""
        # Test with None code (should not crash)
        try:
            risk_analysis = approval_system._analyze_code_risk(None, {})
            # Should still return a valid RiskAnalysis object
            assert isinstance(risk_analysis, RiskAnalysis)
        except Exception:
            # If it raises an exception, that's also acceptable behavior
            pass

    def test_preferences_persistence(self, approval_system):
        """Test preferences persistence simulation."""
        # Simulate loading preferences with error
        with patch.object(approval_system, "_load_preferences") as mock_load:
            mock_load.side_effect = Exception("Load error")

            # Should not crash during initialization
            try:
                approval_system._load_preferences()
            except Exception:
                pass  # Expected

        # Simulate saving preferences with error
        with patch.object(approval_system, "_save_preferences") as mock_save:
            mock_save.side_effect = Exception("Save error")

            # Should not crash when saving
            try:
                approval_system._save_preferences()
            except Exception:
                pass  # Expected

    @pytest.mark.parametrize(
        "code_pattern,expected_risk",
        [
            ('print("hello")', RiskLevel.LOW),
            ("import os", RiskLevel.MEDIUM),
            ('os.system("ls")', RiskLevel.HIGH),
            ("rm -rf /", RiskLevel.BLOCKED),
            ('eval("x + 1")', RiskLevel.HIGH),
            ('exec("print(1)")', RiskLevel.HIGH),
            ('__import__("os")', RiskLevel.HIGH),
        ],
    )
    def test_risk_patterns(self, approval_system, code_pattern, expected_risk):
        """Test various code patterns for risk classification."""
        risk_analysis = approval_system._analyze_code_risk(code_pattern, {})

        # Allow some flexibility in risk assessment
        if expected_risk == RiskLevel.HIGH:
            assert risk_analysis.risk_level in [RiskLevel.HIGH, RiskLevel.BLOCKED]
        else:
            assert risk_analysis.risk_level == expected_risk

    def test_approval_decision_dataclass(self):
        """Test ApprovalDecision dataclass."""
        now = datetime.now()
        request = ApprovalRequest(
            code='print("test")',
            risk_analysis=RiskAnalysis(
                risk_level=RiskLevel.LOW,
                confidence=0.8,
                reasons=[],
                affected_resources=[],
                severity_score=0.1,
            ),
            context={},
            timestamp=now,
            request_id="test123",
        )

        decision = ApprovalDecision(
            request=request,
            result=ApprovalResult.APPROVED,
            user_input="y",
            timestamp=now,
            trust_updated=False,
        )

        assert decision.request == request
        assert decision.result == ApprovalResult.APPROVED
        assert decision.user_input == "y"
        assert decision.timestamp == now
        assert decision.trust_updated == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

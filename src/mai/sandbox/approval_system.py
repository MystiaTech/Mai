"""
Risk-based User Approval System

This module provides a sophisticated approval system that evaluates code execution
requests based on risk analysis and provides appropriate user interaction workflows.
"""

import logging
import json
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime
import sys
import re

from ..core.config import get_config


class RiskLevel(Enum):
    """Risk levels for code execution."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    BLOCKED = "blocked"


class ApprovalResult(Enum):
    """Approval decision results."""

    ALLOWED = "allowed"
    DENIED = "denied"
    BLOCKED = "blocked"
    APPROVED = "approved"


@dataclass
class RiskAnalysis:
    """Risk analysis result."""

    risk_level: RiskLevel
    confidence: float
    reasons: List[str]
    affected_resources: List[str]
    severity_score: float


@dataclass
class ApprovalRequest:
    """Approval request data."""

    code: str
    risk_analysis: RiskAnalysis
    context: Dict[str, Any]
    timestamp: datetime
    request_id: str
    user_preference: Optional[str] = None


@dataclass
class ApprovalDecision:
    """Approval decision record."""

    request: ApprovalRequest
    result: ApprovalResult
    user_input: str
    timestamp: datetime
    trust_updated: bool = False


class ApprovalSystem:
    """Risk-based approval system for code execution."""

    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        self.approval_history: List[ApprovalDecision] = []
        self.user_preferences: Dict[str, str] = {}
        self.trust_patterns: Dict[str, int] = {}

        # Risk thresholds - use defaults since sandbox config not yet in main Config
        self.risk_thresholds = {
            "low_threshold": 0.3,
            "medium_threshold": 0.6,
            "high_threshold": 0.8,
        }

        # Load saved preferences
        self._load_preferences()

    def _load_preferences(self):
        """Load user preferences from configuration."""
        try:
            # For now, preferences are stored locally only
            # TODO: Integrate with Config class when sandbox config added
            self.user_preferences = {}
        except Exception as e:
            self.logger.warning(f"Could not load user preferences: {e}")

    def _save_preferences(self):
        """Save user preferences to configuration."""
        try:
            # Note: This would integrate with config hot-reload system
            pass
        except Exception as e:
            self.logger.warning(f"Could not save user preferences: {e}")

    def _generate_request_id(self, code: str) -> str:
        """Generate unique request ID for code."""
        content = f"{code}_{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _analyze_code_risk(self, code: str, context: Dict[str, Any]) -> RiskAnalysis:
        """Analyze code for potential risks."""
        risk_patterns = {
            "HIGH": [
                r"os\.system\s*\(",
                r"subprocess\.call\s*\(",
                r"exec\s*\(",
                r"eval\s*\(",
                r"__import__\s*\(",
                r'open\s*\([\'"]\/',
                r"shutil\.rmtree",
                r"pickle\.loads?",
            ],
            "MEDIUM": [
                r"import\s+os",
                r"import\s+subprocess",
                r"import\s+sys",
                r"open\s*\(",
                r"file\s*\(",
                r"\.write\s*\(",
                r"\.read\s*\(",
            ],
        }

        risk_score = 0.0
        reasons = []
        affected_resources = []

        # Check for high-risk patterns
        for pattern in risk_patterns["HIGH"]:
            if re.search(pattern, code, re.IGNORECASE):
                risk_score += 0.4
                reasons.append(f"High-risk pattern detected: {pattern}")
                affected_resources.append("system_operations")

        # Check for medium-risk patterns
        for pattern in risk_patterns["MEDIUM"]:
            if re.search(pattern, code, re.IGNORECASE):
                risk_score += 0.2
                reasons.append(f"Medium-risk pattern detected: {pattern}")
                affected_resources.append("file_system")

        # Analyze context
        if context.get("user_level") == "new":
            risk_score += 0.1
            reasons.append("New user profile")

        # Determine risk level
        if risk_score >= self.risk_thresholds["high_threshold"]:
            risk_level = RiskLevel.HIGH
        elif risk_score >= self.risk_thresholds["medium_threshold"]:
            risk_level = RiskLevel.MEDIUM
        elif risk_score >= self.risk_thresholds["low_threshold"]:
            risk_level = RiskLevel.LOW
        else:
            risk_level = RiskLevel.LOW  # Default to low for very safe code

        # Check for blocked operations
        blocked_patterns = [
            r"rm\s+-rf\s+\/",
            r"dd\s+if=",
            r"format\s+",
            r"fdisk",
        ]

        for pattern in blocked_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                risk_level = RiskLevel.BLOCKED
                reasons.append(f"Blocked operation detected: {pattern}")
                break

        confidence = min(0.95, 0.5 + len(reasons) * 0.1)

        return RiskAnalysis(
            risk_level=risk_level,
            confidence=confidence,
            reasons=reasons,
            affected_resources=affected_resources,
            severity_score=risk_score,
        )

    def _present_approval_request(self, request: ApprovalRequest) -> str:
        """Present approval request to user based on risk level."""
        risk_level = request.risk_analysis.risk_level

        if risk_level == RiskLevel.LOW:
            return self._present_low_risk_request(request)
        elif risk_level == RiskLevel.MEDIUM:
            return self._present_medium_risk_request(request)
        elif risk_level == RiskLevel.HIGH:
            return self._present_high_risk_request(request)
        else:  # BLOCKED
            return self._present_blocked_request(request)

    def _present_low_risk_request(self, request: ApprovalRequest) -> str:
        """Present low-risk approval request."""
        print(f"\n🟢 [LOW RISK] Execute {self._get_operation_type(request.code)}?")
        print(f"Code: {request.code[:100]}{'...' if len(request.code) > 100 else ''}")

        response = input("Allow? [Y/n/a(llow always)]: ").strip().lower()

        if response in ["", "y", "yes"]:
            return "allowed"
        elif response == "a":
            self.user_preferences[self._get_operation_type(request.code)] = "auto_allow"
            return "allowed_always"
        else:
            return "denied"

    def _present_medium_risk_request(self, request: ApprovalRequest) -> str:
        """Present medium-risk approval request with details."""
        print(f"\n🟡 [MEDIUM RISK] Potentially dangerous operation detected")
        print(f"Operation Type: {self._get_operation_type(request.code)}")
        print(f"Affected Resources: {', '.join(request.risk_analysis.affected_resources)}")
        print(f"Risk Factors: {len(request.risk_analysis.reasons)}")
        print(f"\nCode Preview:")
        print(request.code[:200] + ("..." if len(request.code) > 200 else ""))

        if request.risk_analysis.reasons:
            print(f"\nRisk Reasons:")
            for reason in request.risk_analysis.reasons[:3]:
                print(f"  • {reason}")

        response = input("\nAllow this operation? [y/N/d(etails)/a(llow always)]: ").strip().lower()

        if response == "y":
            return "allowed"
        elif response == "d":
            return self._present_detailed_view(request)
        elif response == "a":
            self.user_preferences[self._get_operation_type(request.code)] = "auto_allow"
            return "allowed_always"
        else:
            return "denied"

    def _present_high_risk_request(self, request: ApprovalRequest) -> str:
        """Present high-risk approval request with full details."""
        print(f"\n🔴 [HIGH RISK] Dangerous operation detected!")
        print(f"Severity Score: {request.risk_analysis.severity_score:.2f}")
        print(f"Confidence: {request.risk_analysis.confidence:.2f}")

        print(f"\nAffected Resources: {', '.join(request.risk_analysis.affected_resources)}")
        print(f"\nAll Risk Factors:")
        for reason in request.risk_analysis.reasons:
            print(f"  • {reason}")

        print(f"\nFull Code:")
        print("=" * 50)
        print(request.code)
        print("=" * 50)

        print(f"\n⚠️  This operation could potentially harm your system or data.")

        response = (
            input("\nType 'confirm' to allow, 'cancel' to deny, 'details' for more info: ")
            .strip()
            .lower()
        )

        if response == "confirm":
            return "allowed"
        elif response == "details":
            return self._present_detailed_analysis(request)
        else:
            return "denied"

    def _present_blocked_request(self, request: ApprovalRequest) -> str:
        """Present blocked operation notification."""
        print(f"\n🚫 [BLOCKED] Operation not permitted")
        print(f"This operation is blocked for security reasons:")
        for reason in request.risk_analysis.reasons:
            print(f"  • {reason}")
        print("\nThis operation cannot be executed.")
        return "blocked"

    def _present_detailed_view(self, request: ApprovalRequest) -> str:
        """Present detailed view of the request."""
        print(f"\n📋 Detailed Analysis")
        print(f"Request ID: {request.request_id}")
        print(f"Timestamp: {request.timestamp}")
        print(f"Risk Level: {request.risk_analysis.risk_level.value.upper()}")
        print(f"Severity Score: {request.risk_analysis.severity_score:.2f}")

        print(f"\nContext Information:")
        for key, value in request.context.items():
            print(f"  {key}: {value}")

        print(f"\nFull Code:")
        print("=" * 50)
        print(request.code)
        print("=" * 50)

        response = input("\nProceed with execution? [y/N]: ").strip().lower()
        return "allowed" if response == "y" else "denied"

    def _present_detailed_analysis(self, request: ApprovalRequest) -> str:
        """Present extremely detailed analysis for high-risk operations."""
        print(f"\n🔬 Security Analysis Report")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Request ID: {request.request_id}")

        print(f"\nRisk Assessment:")
        print(f"  Level: {request.risk_analysis.risk_level.value.upper()}")
        print(f"  Score: {request.risk_analysis.severity_score:.2f}/1.0")
        print(f"  Confidence: {request.risk_analysis.confidence:.2f}")

        print(f"\nThreat Analysis:")
        for reason in request.risk_analysis.reasons:
            print(f"  ⚠️  {reason}")

        print(f"\nResource Impact:")
        for resource in request.risk_analysis.affected_resources:
            print(f"  📁 {resource}")

        print(
            f"\nRecommendation: {'DENY' if request.risk_analysis.severity_score > 0.8 else 'REVIEW CAREFULLY'}"
        )

        response = input("\nFinal decision? [confirm/cancel]: ").strip().lower()
        return "allowed" if response == "confirm" else "denied"

    def _get_operation_type(self, code: str) -> str:
        """Extract operation type from code."""
        if "import" in code:
            return "module_import"
        elif "os.system" in code or "subprocess" in code:
            return "system_command"
        elif "open(" in code:
            return "file_operation"
        elif "print(" in code:
            return "output_operation"
        else:
            return "code_execution"

    def request_approval(
        self, code: str, context: Optional[Dict[str, Any]] = None
    ) -> Tuple[ApprovalResult, Optional[ApprovalDecision]]:
        """Request user approval for code execution."""
        if context is None:
            context = {}

        # Analyze risk
        risk_analysis = self._analyze_code_risk(code, context)

        # Create request
        request = ApprovalRequest(
            code=code,
            risk_analysis=risk_analysis,
            context=context,
            timestamp=datetime.now(),
            request_id=self._generate_request_id(code),
        )

        # Check user preferences
        operation_type = self._get_operation_type(code)
        if (
            self.user_preferences.get(operation_type) == "auto_allow"
            and risk_analysis.risk_level == RiskLevel.LOW
        ):
            decision = ApprovalDecision(
                request=request,
                result=ApprovalResult.ALLOWED,
                user_input="auto_allowed",
                timestamp=datetime.now(),
            )
            self.approval_history.append(decision)
            return ApprovalResult.ALLOWED, decision

        # Present request based on risk level
        user_response = self._present_approval_request(request)

        # Convert user response to approval result
        if user_response == "blocked":
            result = ApprovalResult.BLOCKED
        elif user_response in ["allowed", "allowed_always"]:
            result = ApprovalResult.APPROVED
        else:
            result = ApprovalResult.DENIED

        # Create decision record
        decision = ApprovalDecision(
            request=request,
            result=result,
            user_input=user_response,
            timestamp=datetime.now(),
            trust_updated=("allowed_always" in user_response),
        )

        # Save decision
        self.approval_history.append(decision)
        if decision.trust_updated:
            self._save_preferences()

        return result, decision

    def get_approval_history(self, limit: int = 10) -> List[ApprovalDecision]:
        """Get recent approval history."""
        return self.approval_history[-limit:]

    def get_trust_patterns(self) -> Dict[str, int]:
        """Get learned trust patterns."""
        patterns = {}
        for decision in self.approval_history:
            op_type = self._get_operation_type(decision.request.code)
            if decision.result == ApprovalResult.APPROVED:
                patterns[op_type] = patterns.get(op_type, 0) + 1
        return patterns

    def reset_preferences(self):
        """Reset all user preferences."""
        self.user_preferences.clear()
        self._save_preferences()

    def is_code_safe(self, code: str) -> bool:
        """Quick check if code is considered safe (no approval needed)."""
        risk_analysis = self._analyze_code_risk(code, {})
        return risk_analysis.risk_level == RiskLevel.LOW and len(risk_analysis.reasons) == 0

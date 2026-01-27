"""Security assessment engine using Bandit and Semgrep."""

import enum
import json
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class SecurityLevel(enum.Enum):
    """Security assessment levels."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    BLOCKED = "BLOCKED"


class SecurityAssessor:
    """Multi-level security assessment using Bandit and Semgrep."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize security assessor with optional configuration."""
        self.config_path = config_path or "config/security.yaml"
        self._load_policies()

    def _load_policies(self) -> None:
        """Load security assessment policies from configuration."""
        # Default policies - will be overridden by config file if exists
        self.policies = {
            "blocked_patterns": [
                "os.system",
                "subprocess.call",
                "eval(",
                "exec(",
                "__import__",
                "open(",
                "file(",
                "input(",
            ],
            "high_triggers": [
                "admin",
                "root",
                "sudo",
                "passwd",
                "shadow",
                "system32",
                "/etc/passwd",
                "/etc/shadow",
            ],
            "thresholds": {"blocked_score": 10, "high_score": 7, "medium_score": 4},
        }

        # Load config file if exists
        config_file = Path(self.config_path)
        if config_file.exists():
            try:
                import yaml

                with open(config_file, "r") as f:
                    config = yaml.safe_load(f)
                    self.policies.update(config.get("policies", {}))
            except Exception:
                # Fall back to defaults if config loading fails
                pass

    def assess(self, code: str) -> Tuple[SecurityLevel, Dict]:
        """Assess code security using Bandit and Semgrep.

        Args:
            code: Python code to analyze

        Returns:
            Tuple of (SecurityLevel, detailed_findings)
        """
        if not code or not code.strip():
            return SecurityLevel.LOW, {"message": "Empty code provided"}

        findings = {
            "bandit_results": [],
            "semgrep_results": [],
            "custom_analysis": {},
            "security_score": 0,
            "recommendations": [],
        }

        try:
            # Run Bandit analysis
            bandit_findings = self._run_bandit(code)
            findings["bandit_results"] = bandit_findings

            # Run Semgrep analysis
            semgrep_findings = self._run_semgrep(code)
            findings["semgrep_results"] = semgrep_findings

            # Custom pattern analysis
            custom_findings = self._analyze_custom_patterns(code)
            findings["custom_analysis"] = custom_findings

            # Calculate security level
            security_level, score = self._calculate_security_level(findings)
            findings["security_score"] = score

            # Generate recommendations
            findings["recommendations"] = self._generate_recommendations(findings)

        except Exception as e:
            # If analysis fails, be conservative
            return SecurityLevel.HIGH, {
                "error": f"Security analysis failed: {str(e)}",
                "fallback_level": "HIGH",
            }

        return security_level, findings

    def _run_bandit(self, code: str) -> List[Dict]:
        """Run Bandit security analysis on code."""
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                temp_file = f.name

            # Run Bandit with JSON output
            result = subprocess.run(
                ["bandit", "-f", "json", temp_file],
                capture_output=True,
                text=True,
                timeout=30,
            )

            # Clean up temp file
            Path(temp_file).unlink(missing_ok=True)

            if result.returncode == 0 or result.returncode == 1:  # 1 means issues found
                try:
                    bandit_output = json.loads(result.stdout)
                    return bandit_output.get("results", [])
                except json.JSONDecodeError:
                    return []
            else:
                return []

        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            # Bandit not available or failed
            return []

    def _run_semgrep(self, code: str) -> List[Dict]:
        """Run Semgrep security analysis on code."""
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                temp_file = f.name

            # Run Semgrep with Python rules
            result = subprocess.run(
                ["semgrep", "--config=p/python", "--json", temp_file],
                capture_output=True,
                text=True,
                timeout=30,
            )

            # Clean up temp file
            Path(temp_file).unlink(missing_ok=True)

            if result.returncode == 0 or result.returncode == 1:  # 1 means findings
                try:
                    semgrep_output = json.loads(result.stdout)
                    return semgrep_output.get("results", [])
                except json.JSONDecodeError:
                    return []
            else:
                return []

        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            # Semgrep not available or failed
            return []

    def _analyze_custom_patterns(self, code: str) -> Dict:
        """Analyze code for custom security patterns."""
        custom_findings = {
            "blocked_patterns": [],
            "high_risk_patterns": [],
            "suspicious_imports": [],
        }

        code_lower = code.lower()
        lines = code.split("\n")

        # Check for blocked patterns
        for i, line in enumerate(lines, 1):
            for pattern in self.policies["blocked_patterns"]:
                if pattern in line:
                    custom_findings["blocked_patterns"].append(
                        {"line": i, "pattern": pattern, "content": line.strip()}
                    )

        # Check for high-risk patterns
        for i, line in enumerate(lines, 1):
            for trigger in self.policies["high_triggers"]:
                if trigger in code_lower and trigger in line.lower():
                    custom_findings["high_risk_patterns"].append(
                        {"line": i, "trigger": trigger, "content": line.strip()}
                    )

        # Check for suspicious imports
        import_keywords = [
            "import os",
            "import sys",
            "import subprocess",
            "import socket",
        ]
        for keyword in import_keywords:
            if keyword in code_lower:
                custom_findings["suspicious_imports"].append(keyword)

        return custom_findings

    def _calculate_security_level(self, findings: Dict) -> Tuple[SecurityLevel, int]:
        """Calculate security level based on all findings."""
        score = 0

        # Score Bandit findings
        for result in findings.get("bandit_results", []):
            severity = result.get("issue_severity", "LOW").upper()
            if severity == "HIGH":
                score += 3
            elif severity == "MEDIUM":
                score += 2
            else:
                score += 1

        # Score Semgrep findings
        for result in findings.get("semgrep_results", []):
            # Semgrep uses different severity levels
            metadata = result.get("metadata", {})
            severity = metadata.get("severity", "INFO").upper()
            if severity == "ERROR":
                score += 3
            elif severity == "WARNING":
                score += 2
            else:
                score += 1

        # Score custom findings
        custom = findings.get("custom_analysis", {})
        score += len(custom.get("blocked_patterns", [])) * 5
        score += len(custom.get("high_risk_patterns", [])) * 3
        score += len(custom.get("suspicious_imports", [])) * 1

        # Determine security level
        thresholds = self.policies["thresholds"]
        if score >= thresholds["blocked_score"]:
            return SecurityLevel.BLOCKED, score
        elif score >= thresholds["high_score"]:
            return SecurityLevel.HIGH, score
        elif score >= thresholds["medium_score"]:
            return SecurityLevel.MEDIUM, score
        else:
            return SecurityLevel.LOW, score

    def _generate_recommendations(self, findings: Dict) -> List[str]:
        """Generate security recommendations based on findings."""
        recommendations = []

        # Analyze Bandit findings
        for result in findings.get("bandit_results", []):
            test_name = result.get("test_name", "")
            if "hardcoded" in test_name.lower():
                recommendations.append("Remove hardcoded credentials or secrets")
            elif "shell" in test_name.lower():
                recommendations.append(
                    "Avoid shell command execution, use safer alternatives"
                )
            elif "pickle" in test_name.lower():
                recommendations.append("Avoid using pickle for untrusted data")

        # Analyze custom findings
        custom = findings.get("custom_analysis", {})
        if custom.get("blocked_patterns"):
            recommendations.append("Remove or sanitize dangerous function calls")
        if custom.get("high_risk_patterns"):
            recommendations.append("Review and justify high-risk system operations")
        if custom.get("suspicious_imports"):
            recommendations.append("Validate necessity of system-level imports")

        if not recommendations:
            recommendations.append("Code appears safe from common security issues")

        return recommendations

"""
Risk Analysis for Mai Sandbox System

Provides AST-based code analysis to detect dangerous patterns
and calculate risk scores for code execution decisions.
"""

import ast
import re
from dataclasses import dataclass


@dataclass
class RiskPattern:
    """Represents a detected risky code pattern"""

    pattern: str
    severity: str  # 'BLOCKED', 'HIGH', 'MEDIUM', 'LOW'
    score: int
    line_number: int
    description: str


@dataclass
class RiskAssessment:
    """Result of risk analysis"""

    score: int
    patterns: list[RiskPattern]
    safe_to_execute: bool
    approval_required: bool


class RiskAnalyzer:
    """
    Analyzes code for dangerous patterns using AST parsing
    and static analysis techniques.
    """

    # Severity scores and risk thresholds
    SEVERITY_SCORES = {"BLOCKED": 100, "HIGH": 80, "MEDIUM": 50, "LOW": 20}

    # Known dangerous patterns
    DANGEROUS_IMPORTS = {
        "os.system": ("BLOCKED", "Direct system command execution"),
        "os.popen": ("BLOCKED", "Direct system command execution"),
        "subprocess.run": ("HIGH", "Subprocess execution"),
        "subprocess.call": ("HIGH", "Subprocess execution"),
        "subprocess.Popen": ("HIGH", "Subprocess execution"),
        "eval": ("HIGH", "Dynamic code execution"),
        "exec": ("HIGH", "Dynamic code execution"),
        "compile": ("MEDIUM", "Code compilation"),
        "__import__": ("MEDIUM", "Dynamic import"),
        "open": ("LOW", "File access"),
        "shutil.rmtree": ("HIGH", "Directory deletion"),
        "os.remove": ("HIGH", "File deletion"),
        "os.unlink": ("HIGH", "File deletion"),
        "os.mkdir": ("LOW", "Directory creation"),
        "os.chdir": ("MEDIUM", "Directory change"),
    }

    # Regex patterns for additional checks
    REGEX_PATTERNS = [
        (r"/dev/[^\\s]+", "BLOCKED", "Device file access"),
        (r"rm\\s+-rf\\s+/", "BLOCKED", "Recursive root deletion"),
        (r"shell=True", "HIGH", "Shell execution in subprocess"),
        (r"password", "MEDIUM", "Potential password handling"),
        (r"api[_-]?key", "MEDIUM", "Potential API key handling"),
        (r"chmod\\s+777", "HIGH", "Permissive file permissions"),
        (r"sudo\\s+", "HIGH", "Privilege escalation"),
    ]

    def __init__(self):
        """Initialize risk analyzer"""
        self.reset_analysis()

    def reset_analysis(self):
        """Reset analysis state"""
        self.detected_patterns: list[RiskPattern] = []

    def analyze_ast(self, code: str) -> RiskAssessment:
        """
        Analyze Python code using AST parsing

        Args:
            code: Python source code to analyze

        Returns:
            RiskAssessment with score, patterns, and execution decision
        """
        self.reset_analysis()

        try:
            tree = ast.parse(code)
            self._walk_ast(tree)
        except SyntaxError as e:
            # Syntax errors are automatically high risk
            pattern = RiskPattern(
                pattern="syntax_error",
                severity="HIGH",
                score=90,
                line_number=getattr(e, "lineno", 0),
                description=f"Syntax error: {e}",
            )
            self.detected_patterns.append(pattern)

        # Additional regex-based checks
        self._regex_checks(code)

        # Calculate overall assessment
        total_score = max([p.score for p in self.detected_patterns] + [0])

        assessment = RiskAssessment(
            score=total_score,
            patterns=self.detected_patterns.copy(),
            safe_to_execute=total_score < 50,
            approval_required=total_score >= 30,
        )

        return assessment

    def detect_dangerous_patterns(self, code: str) -> list[RiskPattern]:
        """
        Detect dangerous patterns using both AST and regex analysis

        Args:
            code: Python source code

        Returns:
            List of detected RiskPattern objects
        """
        assessment = self.analyze_ast(code)
        return assessment.patterns

    def calculate_risk_score(self, patterns: list[RiskPattern]) -> int:
        """
        Calculate overall risk score from detected patterns

        Args:
            patterns: List of detected risk patterns

        Returns:
            Overall risk score (0-100)
        """
        if not patterns:
            return 0

        return max([p.score for p in patterns])

    def _walk_ast(self, tree: ast.AST):
        """Walk AST tree and detect dangerous patterns"""
        for node in ast.walk(tree):
            self._check_imports(node)
            self._check_function_calls(node)
            self._check_file_operations(node)

    def _check_imports(self, node: ast.AST):
        """Check for dangerous imports"""
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name in self.DANGEROUS_IMPORTS:
                    severity, desc = self.DANGEROUS_IMPORTS[name]
                    pattern = RiskPattern(
                        pattern=f"import_{name}",
                        severity=severity,
                        score=self.SEVERITY_SCORES[severity],
                        line_number=getattr(node, "lineno", 0),
                        description=f"Import of {desc}",
                    )
                    self.detected_patterns.append(pattern)

        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module in self.DANGEROUS_IMPORTS:
                name = node.module
                severity, desc = self.DANGEROUS_IMPORTS[name]
                pattern = RiskPattern(
                    pattern=f"from_{name}",
                    severity=severity,
                    score=self.SEVERITY_SCORES[severity],
                    line_number=getattr(node, "lineno", 0),
                    description=f"Import from {desc}",
                )
                self.detected_patterns.append(pattern)

    def _check_function_calls(self, node: ast.AST):
        """Check for dangerous function calls"""
        if isinstance(node, ast.Call):
            # Get function name
            func_name = self._get_function_name(node.func)
            if func_name in self.DANGEROUS_IMPORTS:
                severity, desc = self.DANGEROUS_IMPORTS[func_name]
                pattern = RiskPattern(
                    pattern=f"call_{func_name}",
                    severity=severity,
                    score=self.SEVERITY_SCORES[severity],
                    line_number=getattr(node, "lineno", 0),
                    description=f"Call to {desc}",
                )
                self.detected_patterns.append(pattern)

            # Check for shell=True in subprocess calls
            if func_name in ["subprocess.run", "subprocess.call", "subprocess.Popen"]:
                for keyword in node.keywords:
                    if keyword.arg == "shell" and isinstance(keyword.value, ast.Constant):
                        if keyword.value.value is True:
                            pattern = RiskPattern(
                                pattern="shell_true",
                                severity="HIGH",
                                score=self.SEVERITY_SCORES["HIGH"],
                                line_number=getattr(node, "lineno", 0),
                                description="Shell execution in subprocess",
                            )
                            self.detected_patterns.append(pattern)

    def _check_file_operations(self, node: ast.AST):
        """Check for dangerous file operations"""
        if isinstance(node, ast.Call):
            func_name = self._get_function_name(node.func)
            dangerous_file_ops = ["shutil.rmtree", "os.remove", "os.unlink", "os.chmod", "os.chown"]
            if func_name in dangerous_file_ops:
                severity = "HIGH" if "rmtree" in func_name else "MEDIUM"
                pattern = RiskPattern(
                    pattern=f"file_{func_name}",
                    severity=severity,
                    score=self.SEVERITY_SCORES[severity],
                    line_number=getattr(node, "lineno", 0),
                    description=f"Dangerous file operation: {func_name}",
                )
                self.detected_patterns.append(pattern)

    def _get_function_name(self, node: ast.AST) -> str:
        """Extract function name from AST node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            attr = []
            while isinstance(node, ast.Attribute):
                attr.append(node.attr)
                node = node.value
            if isinstance(node, ast.Name):
                attr.append(node.id)
            return ".".join(reversed(attr))
        return ""

    def _regex_checks(self, code: str):
        """Perform regex-based pattern detection"""
        lines = code.split("\\n")

        for pattern_str, severity, description in self.REGEX_PATTERNS:
            for line_num, line in enumerate(lines, 1):
                if re.search(pattern_str, line, re.IGNORECASE):
                    pattern = RiskPattern(
                        pattern=pattern_str,
                        severity=severity,
                        score=self.SEVERITY_SCORES[severity],
                        line_number=line_num,
                        description=f"Regex detected: {description}",
                    )
                    self.detected_patterns.append(pattern)

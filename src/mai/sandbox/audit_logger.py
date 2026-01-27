"""
Audit Logging for Mai Sandbox System

Provides immutable, append-only logging with sensitive data masking
and tamper detection for sandbox execution audit trails.
"""

import gzip
import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class AuditEntry:
    """Single audit log entry"""

    timestamp: str
    execution_id: str
    code_hash: str
    risk_score: int
    patterns_detected: list[str]
    execution_result: dict[str, Any]
    resource_usage: dict[str, Any] | None = None
    masked_data: dict[str, str] | None = None
    integrity_hash: str | None = None


@dataclass
class LogIntegrity:
    """Log integrity verification result"""

    is_valid: bool
    tampered_entries: list[int]
    hash_chain_valid: bool
    last_verified: str


class AuditLogger:
    """
    Provides immutable audit logging with sensitive data masking
    and tamper detection for sandbox execution tracking.
    """

    # Patterns for sensitive data masking
    SENSITIVE_PATTERNS = [
        (r"\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b", "[EMAIL_REDACTED]"),
        (r"\\b(?:\\d{1,3}\\.){3}\\d{1,3}\\b", "[IP_REDACTED]"),
        (r"password[\\s]*[:=][\\s]*[^\\s]+", "password=[PASSWORD_REDACTED]"),
        (r"api[_-]?key[\\s]*[:=][\\s]*[^\\s]+", "api_key=[API_KEY_REDACTED]"),
        (r"token[\\s]*[:=][\\s]*[^\\s]+", "token=[TOKEN_REDACTED]"),
        (r"secret[\\s]*[:=][\\s]*[^\\s]+", "secret=[SECRET_REDACTED]"),
        (r"bearers?\\s+[^\\s]+", "Bearer [TOKEN_REDACTED]"),
        (r"\\b(?:\\d{4}[-\\s]?){3}\\d{4}\\b", "[CREDIT_CARD_REDACTED]"),  # Basic CC pattern
        (r"\\b\\d{3}-?\\d{2}-?\\d{4}\\b", "[SSN_REDACTED]"),
    ]

    def __init__(self, log_dir: str | None = None, max_file_size_mb: int = 100):
        """
        Initialize audit logger

        Args:
            log_dir: Directory for log files (default: .mai/logs)
            max_file_size_mb: Maximum file size before rotation
        """
        self.log_dir = Path(log_dir or ".mai/logs")
        self.max_file_size = max_file_size_mb * 1024 * 1024  # Convert to bytes
        self.current_log_file = None
        self.previous_hash = None

        # Ensure log directory exists with secure permissions
        self.log_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.log_dir, 0o700)  # Only owner can access

        # Initialize log file
        self._initialize_log_file()

    def _initialize_log_file(self):
        """Initialize or find current log file"""
        timestamp = datetime.now().strftime("%Y%m%d")
        self.current_log_file = self.log_dir / f"sandbox_audit_{timestamp}.jsonl"

        # Create file if doesn't exist
        if not self.current_log_file.exists():
            self.current_log_file.touch()
            os.chmod(self.current_log_file, 0o600)  # Read/write for owner only

        # Load previous hash for integrity chain
        self.previous_hash = self._get_last_hash()

    def log_execution(
        self,
        code: str,
        execution_result: dict[str, Any],
        risk_assessment: dict[str, Any] | None = None,
        resource_usage: dict[str, Any] | None = None,
    ) -> str:
        """
        Log code execution with full audit trail

        Args:
            code: Executed code string
            execution_result: Result of execution
            risk_assessment: Risk analysis results
            resource_usage: Resource usage during execution

        Returns:
            Execution ID for this log entry
        """
        # Generate execution ID and timestamp
        execution_id = hashlib.sha256(f"{time.time()}{code[:100]}".encode()).hexdigest()[:16]
        timestamp = datetime.now().isoformat()

        # Calculate code hash
        code_hash = hashlib.sha256(code.encode()).hexdigest()

        # Extract risk information
        risk_score = 0
        patterns_detected = []
        if risk_assessment:
            risk_score = risk_assessment.get("score", 0)
            patterns_detected = [p.get("pattern", "") for p in risk_assessment.get("patterns", [])]

        # Mask sensitive data in code
        masked_code, masked_info = self.mask_sensitive_data(code)

        # Create audit entry
        entry = AuditEntry(
            timestamp=timestamp,
            execution_id=execution_id,
            code_hash=code_hash,
            risk_score=risk_score,
            patterns_detected=patterns_detected,
            execution_result=execution_result,
            resource_usage=resource_usage,
            masked_data=masked_info,
            integrity_hash=None,  # Will be calculated
        )

        # Calculate integrity hash with previous hash
        entry.integrity_hash = self._calculate_chain_hash(entry)

        # Write to log file
        self._write_entry(entry)

        # Check if rotation needed
        if self.current_log_file.stat().st_size > self.max_file_size:
            self._rotate_logs()

        return execution_id

    def mask_sensitive_data(self, text: str) -> tuple[str, dict[str, str]]:
        """
        Mask sensitive data patterns in text

        Args:
            text: Text to mask

        Returns:
            Tuple of (masked_text, masking_info)
        """
        masked_text = text
        masking_info = {}

        for pattern, replacement in self.SENSITIVE_PATTERNS:
            matches = re.findall(pattern, masked_text, re.IGNORECASE)
            if matches:
                masking_info[pattern] = f"Replaced {len(matches)} instances"
                masked_text = re.sub(pattern, replacement, masked_text, flags=re.IGNORECASE)

        return masked_text, masking_info

    def rotate_logs(self):
        """Rotate current log file with compression"""
        if not self.current_log_file.exists():
            return

        # Compress old log
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        compressed_file = self.log_dir / f"sandbox_audit_{timestamp}.jsonl.gz"

        with open(self.current_log_file, "rb") as f_in:
            with gzip.open(compressed_file, "wb") as f_out:
                f_out.writelines(f_in)

        # Remove original
        self.current_log_file.unlink()

        # Set secure permissions on compressed file
        os.chmod(compressed_file, 0o600)

        # Reinitialize new log file
        self._initialize_log_file()

    def verify_integrity(self) -> LogIntegrity:
        """
        Verify log file integrity using hash chain

        Returns:
            LogIntegrity verification result
        """
        if not self.current_log_file.exists():
            return LogIntegrity(
                is_valid=False,
                tampered_entries=[],
                hash_chain_valid=False,
                last_verified=datetime.now().isoformat(),
            )

        try:
            with open(self.current_log_file) as f:
                lines = f.readlines()

            tampered_entries = []
            previous_hash = None

            for i, line in enumerate(lines):
                try:
                    entry_data = json.loads(line.strip())
                    expected_hash = entry_data.get("integrity_hash")

                    # Recalculate hash without integrity field
                    entry_data["integrity_hash"] = None
                    actual_hash = hashlib.sha256(
                        json.dumps(entry_data, sort_keys=True).encode()
                    ).hexdigest()

                    if previous_hash:
                        # Include previous hash in calculation
                        combined = f"{previous_hash}{actual_hash}"
                        actual_hash = hashlib.sha256(combined.encode()).hexdigest()

                    if expected_hash != actual_hash:
                        tampered_entries.append(i)

                    previous_hash = expected_hash

                except (json.JSONDecodeError, KeyError):
                    tampered_entries.append(i)

            return LogIntegrity(
                is_valid=len(tampered_entries) == 0,
                tampered_entries=tampered_entries,
                hash_chain_valid=len(tampered_entries) == 0,
                last_verified=datetime.now().isoformat(),
            )

        except Exception:
            return LogIntegrity(
                is_valid=False,
                tampered_entries=[],
                hash_chain_valid=False,
                last_verified=datetime.now().isoformat(),
            )

    def query_logs(
        self, limit: int = 100, risk_min: int = 0, after: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Query audit logs with filters

        Args:
            limit: Maximum number of entries to return
            risk_min: Minimum risk score to include
            after: ISO timestamp to filter after

        Returns:
            List of matching log entries
        """
        if not self.current_log_file.exists():
            return []

        entries = []

        try:
            with open(self.current_log_file) as f:
                for line in f:
                    if not line.strip():
                        continue

                    try:
                        entry = json.loads(line.strip())

                        # Apply filters
                        if entry.get("risk_score", 0) < risk_min:
                            continue

                        if after and entry.get("timestamp", "") <= after:
                            continue

                        entries.append(entry)

                        if len(entries) >= limit:
                            break

                    except json.JSONDecodeError:
                        continue

        except Exception:
            return []

        # Return in reverse chronological order
        return list(reversed(entries[-limit:]))

    def get_execution_by_id(self, execution_id: str) -> dict[str, Any] | None:
        """
        Retrieve specific execution by ID

        Args:
            execution_id: Unique execution identifier

        Returns:
            Log entry or None if not found
        """
        entries = self.query_logs(limit=1000)  # Get more for search

        for entry in entries:
            if entry.get("execution_id") == execution_id:
                return entry

        return None

    def _write_entry(self, entry: AuditEntry):
        """Write entry to log file"""
        try:
            with open(self.current_log_file, "a") as f:
                # Convert to dict and remove None values
                entry_dict = {k: v for k, v in asdict(entry).items() if v is not None}
                f.write(json.dumps(entry_dict) + "\\n")
                f.flush()  # Ensure immediate write

            # Update previous hash
            self.previous_hash = entry.integrity_hash

        except Exception as e:
            raise RuntimeError(f"Failed to write audit entry: {e}") from e

    def _calculate_chain_hash(self, entry: AuditEntry) -> str:
        """Calculate integrity hash for entry with previous hash"""
        entry_dict = asdict(entry)
        entry_dict["integrity_hash"] = None  # Exclude from calculation

        # Create hash of entry data
        entry_hash = hashlib.sha256(json.dumps(entry_dict, sort_keys=True).encode()).hexdigest()

        # Chain with previous hash if exists
        if self.previous_hash:
            combined = f"{self.previous_hash}{entry_hash}"
            return hashlib.sha256(combined.encode()).hexdigest()

        return entry_hash

    def _get_last_hash(self) -> str | None:
        """Get hash from last entry in log file"""
        if not self.current_log_file.exists():
            return None

        try:
            with open(self.current_log_file) as f:
                lines = f.readlines()

            if not lines:
                return None

            last_line = lines[-1].strip()
            if not last_line:
                return None

            entry = json.loads(last_line)
            return entry.get("integrity_hash")

        except (json.JSONDecodeError, FileNotFoundError):
            return None

    def _rotate_logs(self):
        """Perform log rotation"""
        try:
            self.rotate_logs()
        except Exception as e:
            print(f"Log rotation failed: {e}")

    def get_log_stats(self) -> dict[str, Any]:
        """
        Get statistics about audit logs

        Returns:
            Dictionary with log statistics
        """
        if not self.current_log_file.exists():
            return {
                "total_entries": 0,
                "file_size_bytes": 0,
                "high_risk_executions": 0,
                "last_execution": None,
            }

        try:
            with open(self.current_log_file) as f:
                lines = f.readlines()

            entries = []
            high_risk_count = 0

            for line in lines:
                if not line.strip():
                    continue

                try:
                    entry = json.loads(line.strip())
                    entries.append(entry)

                    if entry.get("risk_score", 0) >= 70:
                        high_risk_count += 1

                except json.JSONDecodeError:
                    continue

            file_size = self.current_log_file.stat().st_size
            last_execution = entries[-1].get("timestamp") if entries else None

            return {
                "total_entries": len(entries),
                "file_size_bytes": file_size,
                "file_size_mb": file_size / (1024 * 1024),
                "high_risk_executions": high_risk_count,
                "last_execution": last_execution,
                "log_file": str(self.current_log_file),
            }

        except Exception:
            return {
                "total_entries": 0,
                "file_size_bytes": 0,
                "high_risk_executions": 0,
                "last_execution": None,
            }

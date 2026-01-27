"""Tamper-proof logger with SHA-256 hash chains for integrity protection."""

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import threading


class TamperProofLogger:
    """
    Tamper-proof logger using SHA-256 hash chains to detect log tampering.

    Each log entry contains:
    - Timestamp
    - Event type and data
    - Current hash (SHA-256)
    - Previous hash (for chain integrity)
    - Cryptographic signature
    """

    def __init__(self, log_file: Optional[str] = None, storage_dir: str = "logs/audit"):
        """Initialize tamper-proof logger with hash chain."""
        self.log_file = log_file or f"{storage_dir}/audit.log"
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.previous_hash: Optional[str] = None
        self.log_entries: List[Dict] = []
        self.lock = threading.Lock()

        # Initialize hash chain from existing log if present
        self._initialize_hash_chain()

    def _initialize_hash_chain(self) -> None:
        """Load existing log entries and establish hash chain."""
        log_path = Path(self.log_file)
        if log_path.exists():
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            entry = json.loads(line.strip())
                            self.log_entries.append(entry)
                            self.previous_hash = entry.get("hash")
            except (json.JSONDecodeError, IOError):
                # Start fresh if log is corrupted
                self.log_entries = []
                self.previous_hash = None

    def _calculate_hash(
        self, event_data: Dict, previous_hash: Optional[str] = None
    ) -> str:
        """
        Calculate SHA-256 hash for event data and previous hash.

        Args:
            event_data: Event data to hash
            previous_hash: Previous hash in chain

        Returns:
            SHA-256 hash as hex string
        """
        # Create canonical JSON representation
        canonical_data = {
            "timestamp": event_data.get("timestamp"),
            "event_type": event_data.get("event_type"),
            "event_data": event_data.get("event_data"),
            "previous_hash": previous_hash,
        }

        # Sort keys for consistent hashing
        json_str = json.dumps(canonical_data, sort_keys=True, separators=(",", ":"))

        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()

    def _sign_hash(self, hash_value: str) -> str:
        """
        Create cryptographic signature for hash value.

        Args:
            hash_value: Hash to sign

        Returns:
            Signature as hex string (simplified implementation)
        """
        # In production, use proper asymmetric cryptography
        # For now, use HMAC with a secret key
        secret_key = "mai-audit-secret-key-change-in-production"
        return hashlib.sha256((hash_value + secret_key).encode("utf-8")).hexdigest()

    def log_event(
        self, event_type: str, event_data: Dict, metadata: Optional[Dict] = None
    ) -> str:
        """
        Log an event with tamper-proof hash chain.

        Args:
            event_type: Type of event (e.g., 'code_execution', 'security_assessment')
            event_data: Event-specific data
            metadata: Optional metadata (e.g., user_id, session_id)

        Returns:
            Current hash of the logged entry
        """
        with self.lock:
            timestamp = datetime.now().isoformat()

            # Prepare event data
            log_entry_data = {
                "timestamp": timestamp,
                "event_type": event_type,
                "event_data": event_data,
                "metadata": metadata or {},
            }

            # Calculate current hash
            current_hash = self._calculate_hash(log_entry_data, self.previous_hash)

            # Create signature
            signature = self._sign_hash(current_hash)

            # Create complete log entry
            log_entry = {
                "timestamp": timestamp,
                "event_type": event_type,
                "event_data": event_data,
                "metadata": metadata or {},
                "hash": current_hash,
                "previous_hash": self.previous_hash,
                "signature": signature,
            }

            # Add to in-memory log
            self.log_entries.append(log_entry)
            self.previous_hash = current_hash

            # Write to file
            self._write_to_file(log_entry)

            return current_hash

    def _write_to_file(self, log_entry: Dict) -> None:
        """Write log entry to file."""
        try:
            log_path = Path(self.log_file)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        except IOError as e:
            # In production, implement proper error handling and backup
            print(f"Warning: Failed to write to audit log: {e}")

    def verify_chain(self) -> Dict[str, Any]:
        """
        Verify the integrity of the entire hash chain.

        Returns:
            Dictionary with verification results
        """
        results = {
            "is_valid": True,
            "total_entries": len(self.log_entries),
            "tampered_entries": [],
            "broken_links": [],
        }

        if not self.log_entries:
            return results

        previous_hash = None

        for i, entry in enumerate(self.log_entries):
            # Recalculate hash
            entry_data = {
                "timestamp": entry.get("timestamp"),
                "event_type": entry.get("event_type"),
                "event_data": entry.get("event_data"),
                "previous_hash": previous_hash,
            }

            calculated_hash = self._calculate_hash(entry_data, previous_hash)
            stored_hash = entry.get("hash")

            if calculated_hash != stored_hash:
                results["is_valid"] = False
                results["tampered_entries"].append(
                    {
                        "entry_index": i,
                        "timestamp": entry.get("timestamp"),
                        "stored_hash": stored_hash,
                        "calculated_hash": calculated_hash,
                    }
                )

            # Check hash chain continuity
            if previous_hash and entry.get("previous_hash") != previous_hash:
                results["is_valid"] = False
                results["broken_links"].append(
                    {
                        "entry_index": i,
                        "timestamp": entry.get("timestamp"),
                        "expected_previous": previous_hash,
                        "actual_previous": entry.get("previous_hash"),
                    }
                )

            # Verify signature
            stored_signature = entry.get("signature")
            if stored_signature:
                expected_signature = self._sign_hash(stored_hash)
                if stored_signature != expected_signature:
                    results["is_valid"] = False
                    results["tampered_entries"].append(
                        {
                            "entry_index": i,
                            "timestamp": entry.get("timestamp"),
                            "issue": "Invalid signature",
                        }
                    )

            previous_hash = stored_hash

        return results

    def get_logs(
        self,
        limit: Optional[int] = None,
        event_type: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> List[Dict]:
        """
        Retrieve logs with optional filtering.

        Args:
            limit: Maximum number of entries to return
            event_type: Filter by event type
            start_time: ISO format timestamp start
            end_time: ISO format timestamp end

        Returns:
            List of log entries
        """
        filtered_logs = self.log_entries.copy()

        # Filter by event type
        if event_type:
            filtered_logs = [
                log for log in filtered_logs if log.get("event_type") == event_type
            ]

        # Filter by time range
        if start_time:
            filtered_logs = [
                log for log in filtered_logs if log.get("timestamp", "") >= start_time
            ]

        if end_time:
            filtered_logs = [
                log for log in filtered_logs if log.get("timestamp", "") <= end_time
            ]

        # Apply limit
        if limit:
            filtered_logs = filtered_logs[-limit:]

        return filtered_logs

    def get_chain_info(self) -> Dict[str, Any]:
        """
        Get information about the hash chain.

        Returns:
            Dictionary with chain statistics
        """
        if not self.log_entries:
            return {
                "total_entries": 0,
                "current_hash": None,
                "first_entry": None,
                "last_entry": None,
                "chain_length": 0,
            }

        return {
            "total_entries": len(self.log_entries),
            "current_hash": self.previous_hash,
            "first_entry": {
                "timestamp": self.log_entries[0].get("timestamp"),
                "hash": self.log_entries[0].get("hash"),
            },
            "last_entry": {
                "timestamp": self.log_entries[-1].get("timestamp"),
                "hash": self.log_entries[-1].get("hash"),
            },
            "chain_length": len(self.log_entries),
        }

    def export_logs(self, output_file: str, include_integrity: bool = True) -> bool:
        """
        Export logs to a file with optional integrity verification.

        Args:
            output_file: Path to output file
            include_integrity: Whether to include verification results

        Returns:
            True if export successful
        """
        try:
            export_data = {
                "logs": self.log_entries,
                "export_timestamp": datetime.now().isoformat(),
            }

            if include_integrity:
                export_data["integrity"] = self.verify_chain()
                export_data["chain_info"] = self.get_chain_info()

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2)

            return True
        except (IOError, json.JSONEncodeError):
            return False

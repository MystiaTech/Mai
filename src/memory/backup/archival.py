"""
JSON archival system for long-term conversation storage.

Provides export/import functionality for compressed conversations
with organized directory structure and version compatibility.
"""

import json
import os
import shutil
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Iterator
from pathlib import Path
import gzip

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from memory.storage.compression import CompressionEngine, CompressedConversation


class ArchivalManager:
    """
    JSON archival manager for compressed conversations.

    Handles export/import of conversations with organized directory
    structure and version compatibility for future upgrades.
    """

    ARCHIVAL_VERSION = "1.0"

    def __init__(
        self,
        archival_root: str = "archive",
        compression_engine: Optional[CompressionEngine] = None,
    ):
        """
        Initialize archival manager.

        Args:
            archival_root: Root directory for archived conversations
            compression_engine: Optional compression engine instance
        """
        self.archival_root = Path(archival_root)
        self.archival_root.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.compression_engine = compression_engine or CompressionEngine()

        # Create archive directory structure
        self._initialize_directory_structure()

    def _initialize_directory_structure(self) -> None:
        """Create standard archive directory structure."""
        # Year/month structure: archive/YYYY/MM/
        for year_dir in self.archival_root.iterdir():
            if year_dir.is_dir() and year_dir.name.isdigit():
                for month in range(1, 13):
                    month_dir = year_dir / f"{month:02d}"
                    month_dir.mkdir(exist_ok=True)

        self.logger.debug(
            f"Archive directory structure initialized: {self.archival_root}"
        )

    def _get_archive_path(self, conversation_date: datetime) -> Path:
        """
        Get archive path for a conversation date.

        Args:
            conversation_date: Date of the conversation

        Returns:
            Path where conversation should be archived
        """
        year_dir = self.archival_root / str(conversation_date.year)
        month_dir = year_dir / f"{conversation_date.month:02d}"

        # Create directories if they don't exist
        year_dir.mkdir(exist_ok=True)
        month_dir.mkdir(exist_ok=True)

        return month_dir

    def archive_conversation(
        self, conversation: Dict[str, Any], compressed: CompressedConversation
    ) -> str:
        """
        Archive a conversation to JSON file.

        Args:
            conversation: Original conversation data
            compressed: Compressed conversation data

        Returns:
            Path to archived file
        """
        try:
            # Get archive path based on conversation date
            conv_date = datetime.fromisoformat(
                conversation.get("created_at", datetime.now().isoformat())
            )
            archive_path = self._get_archive_path(conv_date)

            # Create filename
            timestamp = conv_date.strftime("%Y%m%d_%H%M%S")
            safe_title = "".join(
                c
                for c in conversation.get("title", "untitled")
                if c.isalnum() or c in "-_"
            )[:50]
            filename = f"{timestamp}_{safe_title}_{conversation.get('id', 'unknown')[:8]}.json.gz"
            file_path = archive_path / filename

            # Prepare archival data
            archival_data = {
                "version": self.ARCHIVAL_VERSION,
                "archived_at": datetime.now().isoformat(),
                "original_conversation": conversation,
                "compressed_conversation": {
                    "original_id": compressed.original_id,
                    "compression_level": compressed.compression_level.value,
                    "compressed_at": compressed.compressed_at.isoformat(),
                    "original_created_at": compressed.original_created_at.isoformat(),
                    "content": compressed.content,
                    "metadata": compressed.metadata,
                    "metrics": {
                        "original_length": compressed.metrics.original_length,
                        "compressed_length": compressed.metrics.compressed_length,
                        "compression_ratio": compressed.metrics.compression_ratio,
                        "information_retention_score": compressed.metrics.information_retention_score,
                        "quality_score": compressed.metrics.quality_score,
                    },
                },
            }

            # Write compressed JSON file
            with gzip.open(file_path, "wt", encoding="utf-8") as f:
                json.dump(archival_data, f, indent=2, ensure_ascii=False)

            self.logger.info(
                f"Archived conversation {conversation.get('id')} to {file_path}"
            )
            return str(file_path)

        except Exception as e:
            self.logger.error(
                f"Failed to archive conversation {conversation.get('id')}: {e}"
            )
            raise

    def archive_conversations_batch(
        self, conversations: List[Dict[str, Any]], compress: bool = True
    ) -> List[str]:
        """
        Archive multiple conversations efficiently.

        Args:
            conversations: List of conversations to archive
            compress: Whether to compress conversations before archiving

        Returns:
            List of archived file paths
        """
        archived_paths = []

        for conversation in conversations:
            try:
                # Compress if requested
                if compress:
                    compressed = self.compression_engine.compress_by_age(conversation)
                else:
                    # Create uncompressed version
                    from memory.storage.compression import (
                        CompressionLevel,
                        CompressedConversation,
                        CompressionMetrics,
                    )
                    from datetime import datetime

                    compressed = CompressedConversation(
                        original_id=conversation.get("id", "unknown"),
                        compression_level=CompressionLevel.FULL,
                        compressed_at=datetime.now(),
                        original_created_at=datetime.fromisoformat(
                            conversation.get("created_at", datetime.now().isoformat())
                        ),
                        content=conversation,
                        metadata={"uncompressed": True},
                        metrics=CompressionMetrics(
                            original_length=len(json.dumps(conversation)),
                            compressed_length=len(json.dumps(conversation)),
                            compression_ratio=1.0,
                            information_retention_score=1.0,
                            quality_score=1.0,
                        ),
                    )

                path = self.archive_conversation(conversation, compressed)
                archived_paths.append(path)

            except Exception as e:
                self.logger.error(
                    f"Failed to archive conversation {conversation.get('id', 'unknown')}: {e}"
                )
                continue

        self.logger.info(
            f"Archived {len(archived_paths)}/{len(conversations)} conversations"
        )
        return archived_paths

    def restore_conversation(self, archive_path: str) -> Optional[Dict[str, Any]]:
        """
        Restore a conversation from archive.

        Args:
            archive_path: Path to archived file

        Returns:
            Restored conversation data or None if failed
        """
        try:
            archive_file = Path(archive_path)
            if not archive_file.exists():
                self.logger.error(f"Archive file not found: {archive_path}")
                return None

            # Read and decompress archive file
            with gzip.open(archive_file, "rt", encoding="utf-8") as f:
                archival_data = json.load(f)

            # Verify version compatibility
            version = archival_data.get("version", "unknown")
            if version != self.ARCHIVAL_VERSION:
                self.logger.warning(
                    f"Archive version {version} may not be compatible with current version {self.ARCHIVAL_VERSION}"
                )

            # Return the original conversation (or decompressed version if preferred)
            original_conversation = archival_data.get("original_conversation")
            compressed_info = archival_data.get("compressed_conversation", {})

            # Add archival metadata to conversation
            original_conversation["_archival_info"] = {
                "archived_at": archival_data.get("archived_at"),
                "archive_path": str(archive_file),
                "compression_level": compressed_info.get("compression_level"),
                "compression_ratio": compressed_info.get("metrics", {}).get(
                    "compression_ratio", 1.0
                ),
                "version": version,
            }

            self.logger.info(f"Restored conversation from {archive_path}")
            return original_conversation

        except Exception as e:
            self.logger.error(
                f"Failed to restore conversation from {archive_path}: {e}"
            )
            return None

    def list_archived(
        self,
        year: Optional[int] = None,
        month: Optional[int] = None,
        include_content: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        List archived conversations with optional filtering.

        Args:
            year: Optional year filter
            month: Optional month filter (1-12)
            include_content: Whether to include conversation content

        Returns:
            List of archived conversation info
        """
        archived_list = []

        try:
            # Determine search path
            search_path = self.archival_root
            if year:
                search_path = search_path / str(year)
                if month:
                    search_path = search_path / f"{month:02d}"

            if not search_path.exists():
                return []

            # Scan for archive files
            for archive_file in search_path.rglob("*.json.gz"):
                try:
                    # Read minimal metadata without loading full content
                    with gzip.open(archive_file, "rt", encoding="utf-8") as f:
                        archival_data = json.load(f)

                    conversation = archival_data.get("original_conversation", {})
                    compressed = archival_data.get("compressed_conversation", {})

                    archive_info = {
                        "id": conversation.get("id"),
                        "title": conversation.get("title"),
                        "created_at": conversation.get("created_at"),
                        "archived_at": archival_data.get("archived_at"),
                        "archive_path": str(archive_file),
                        "compression_level": compressed.get("compression_level"),
                        "compression_ratio": compressed.get("metrics", {}).get(
                            "compression_ratio", 1.0
                        ),
                        "version": archival_data.get("version"),
                    }

                    if include_content:
                        archive_info["original_conversation"] = conversation
                        archive_info["compressed_conversation"] = compressed

                    archived_list.append(archive_info)

                except Exception as e:
                    self.logger.error(
                        f"Failed to read archive file {archive_file}: {e}"
                    )
                    continue

            # Sort by archived date (newest first)
            archived_list.sort(key=lambda x: x.get("archived_at", ""), reverse=True)
            return archived_list

        except Exception as e:
            self.logger.error(f"Failed to list archived conversations: {e}")
            return []

    def delete_archive(self, archive_path: str) -> bool:
        """
        Delete an archived conversation.

        Args:
            archive_path: Path to archived file

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            archive_file = Path(archive_path)
            if archive_file.exists():
                archive_file.unlink()
                self.logger.info(f"Deleted archive: {archive_path}")
                return True
            else:
                self.logger.warning(f"Archive file not found: {archive_path}")
                return False
        except Exception as e:
            self.logger.error(f"Failed to delete archive {archive_path}: {e}")
            return False

    def get_archive_stats(self) -> Dict[str, Any]:
        """
        Get statistics about archived conversations.

        Returns:
            Dictionary with archive statistics
        """
        try:
            total_files = 0
            total_size = 0
            compression_levels = {}
            years = set()

            for archive_file in self.archival_root.rglob("*.json.gz"):
                try:
                    total_files += 1
                    total_size += archive_file.stat().st_size

                    # Extract year from path
                    path_parts = archive_file.parts
                    for i, part in enumerate(path_parts):
                        if part == str(self.archival_root.name) and i + 1 < len(
                            path_parts
                        ):
                            year_part = path_parts[i + 1]
                            if year_part.isdigit():
                                years.add(year_part)
                                break

                    # Read compression level without loading full content
                    with gzip.open(archive_file, "rt", encoding="utf-8") as f:
                        archival_data = json.load(f)
                        compressed = archival_data.get("compressed_conversation", {})
                        level = compressed.get("compression_level", "unknown")
                        compression_levels[level] = compression_levels.get(level, 0) + 1

                except Exception as e:
                    self.logger.error(
                        f"Failed to analyze archive file {archive_file}: {e}"
                    )
                    continue

            return {
                "total_archived_conversations": total_files,
                "total_archive_size_bytes": total_size,
                "total_archive_size_mb": round(total_size / (1024 * 1024), 2),
                "compression_levels": compression_levels,
                "years_with_archives": sorted(list(years)),
                "archive_directory": str(self.archival_root),
            }

        except Exception as e:
            self.logger.error(f"Failed to get archive stats: {e}")
            return {}

    def migrate_archives(self, from_version: str, to_version: str) -> int:
        """
        Migrate archives from one version to another.

        Args:
            from_version: Source version
            to_version: Target version

        Returns:
            Number of archives migrated
        """
        # Placeholder for future migration functionality
        self.logger.info(
            f"Migration from {from_version} to {to_version} not yet implemented"
        )
        return 0

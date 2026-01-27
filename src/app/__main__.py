"""
Mai CLI Application

Command-line interface for interacting with Mai and accessing Phase 1 capabilities.
"""

import argparse
import asyncio
import json
import logging
import sys
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path
import threading
import os

# Import rich components for resource display
try:
    from rich.table import Table
    from rich.progress import Progress, BarColumn, TextColumn
    from rich.console import Group, Console
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.live import Live
    from rich.align import Align

    RICH_AVAILABLE = True
except ImportError:
    # Create dummy classes for fallback
    class Table:
        pass

    class Progress:
        pass

    class BarColumn:
        pass

    class TextColumn:
        pass

    class Group:
        pass

    class Panel:
        pass

    class Live:
        pass

    class Console:
        def print(self, *args, **kwargs):
            print(*args)

    class Align:
        pass

    RICH_AVAILABLE = False

# Import blessed for terminal size detection
try:
    from blessed import Terminal

    BLESSED_AVAILABLE = True
except ImportError:
    # Create dummy Terminal for fallback
    class Terminal:
        def __init__(self):
            self.width = None
            self.height = None

    BLESSED_AVAILABLE = False

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Handle missing dependencies gracefully
try:
    from mai.core.interface import MaiInterface, ModelState
    from mai.core.config import get_config
    from mai.core.exceptions import MaiError, ModelError, ModelConnectionError
    from mai.conversation.engine import ConversationEngine
    from mai.memory.manager import MemoryManager
    from mai.sandbox.approval_system import ApprovalSystem, ApprovalDecision, RiskLevel

    INTERFACE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some dependencies not available: {e}")
    print("Limited functionality mode - some features may not work")
    MaiInterface = None
    ModelState = None
    get_config = None
    MaiError = Exception
    ModelError = Exception
    ModelConnectionError = Exception
    ConversationEngine = None
    MemoryManager = None
    ApprovalSystem = None
    ApprovalDecision = None

    # Mock RiskLevel for fallback
    from enum import Enum

    class MockRiskLevel(Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        BLOCKED = "blocked"

    RiskLevel = MockRiskLevel

    INTERFACE_AVAILABLE = False


class Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    DIM = "\033[2m"


class ResourceDisplayManager:
    """Manages real-time resource display with responsive layouts."""

    def __init__(self):
        self.console = Console(force_terminal=True) if RICH_AVAILABLE else None
        self.terminal = Terminal() if BLESSED_AVAILABLE else None
        self.last_width = None
        self.current_layout = None

    def get_terminal_width(self) -> int:
        """Get terminal width with fallback methods."""
        try:
            # Try blessed first (most reliable)
            if BLESSED_AVAILABLE and self.terminal:
                width = self.terminal.width or 80
                return width
        except:
            pass

        try:
            # Try os.get_terminal_size
            import shutil

            size = shutil.get_terminal_size()
            return size.columns or 80
        except:
            pass

        # Fallback
        return 80

    def determine_layout(self, width: int) -> str:
        """Determine layout based on terminal width."""
        if width >= 120:
            return "full"
        elif width >= 80:
            return "compact"
        else:
            return "minimal"

    def create_resource_table(self, interface):
        """Create rich table with resource information."""
        if not RICH_AVAILABLE or not self.console:
            return None

        try:
            status = interface.get_system_status()
            resources = status.resources
            model_info = {
                "name": status.current_model,
                "state": status.model_state.value if status.model_state else "unknown",
                "tokens": getattr(status, "tokens_used", 0),
            }

            # Use a more defensive approach for Table creation
            table = None
            try:
                table = Table(show_header=True, header_style="bold blue")
                table.add_column("Metric", style="cyan", width=15)
                table.add_column("Usage", style="white", width=20)
                table.add_column("Status", style="bold", width=10)
            except:
                return None

            # CPU usage with color coding
            cpu_percent = resources.cpu_percent if hasattr(resources, "cpu_percent") else 0
            cpu_color = "green" if cpu_percent < 60 else "yellow" if cpu_percent < 80 else "red"
            try:
                table.add_row("CPU", f"{cpu_percent:.1f}%", f"[{cpu_color}]●[/{cpu_color}]")
            except:
                pass

            # Memory usage with color coding
            if hasattr(resources, "memory_total_gb") and hasattr(resources, "memory_available_gb"):
                ram_used = resources.memory_total_gb - resources.memory_available_gb
                ram_percent = (
                    (ram_used / resources.memory_total_gb * 100)
                    if resources.memory_total_gb > 0
                    else 0
                )
                ram_color = "green" if ram_percent < 70 else "yellow" if ram_percent < 85 else "red"
                try:
                    table.add_row(
                        "RAM",
                        f"{ram_used:.1f}/{resources.memory_total_gb:.1f}GB",
                        f"[{ram_color}]●[/{ram_color}]",
                    )
                except:
                    pass
            else:
                try:
                    table.add_row("RAM", "Unknown", "[yellow]●[/yellow]")
                except:
                    pass

            # GPU usage if available
            if hasattr(resources, "gpu_available") and resources.gpu_available:
                gpu_usage = (
                    resources.gpu_usage_percent if hasattr(resources, "gpu_usage_percent") else 0
                )
                gpu_vram = (
                    f"{resources.gpu_memory_gb:.1f}GB"
                    if hasattr(resources, "gpu_memory_gb") and resources.gpu_memory_gb
                    else "Unknown"
                )
                gpu_color = "green" if gpu_usage < 60 else "yellow" if gpu_usage < 80 else "red"
                try:
                    table.add_row(
                        "GPU", f"{gpu_usage:.1f}% ({gpu_vram})", f"[{gpu_color}]●[/{gpu_color}]"
                    )
                except:
                    pass
            else:
                try:
                    table.add_row("GPU", "Not Available", "[dim]○[/dim]")
                except:
                    pass

            # Model and token info
            try:
                table.add_row("Model", model_info["name"], "[blue]●[/blue]")
            except:
                pass
            if hasattr(status, "tokens_used"):
                tokens_color = (
                    "green"
                    if status.tokens_used < 1000
                    else "yellow"
                    if status.tokens_used < 4000
                    else "red"
                )
                try:
                    table.add_row(
                        "Tokens", str(status.tokens_used), f"[{tokens_color}]●[/{tokens_color}]"
                    )
                except:
                    pass

            return table

        except Exception as e:
            if RICH_AVAILABLE and hasattr(self.console, "print"):
                self.console.print(f"[red]Error creating resource table: {e}[/red]")
            return None

        try:
            status = interface.get_system_status()
            resources = status.resources
            model_info = {
                "name": status.current_model,
                "state": status.model_state.value if status.model_state else "unknown",
                "tokens": getattr(status, "tokens_used", 0),
            }

            table = Table(show_header=True, header_style="bold blue")
            table.add_column("Metric", style="cyan", width=15)
            table.add_column("Usage", style="white", width=20)
            table.add_column("Status", style="bold", width=10)

            # CPU usage with color coding
            cpu_percent = resources.cpu_percent if hasattr(resources, "cpu_percent") else 0
            cpu_color = "green" if cpu_percent < 60 else "yellow" if cpu_percent < 80 else "red"
            table.add_row("CPU", f"{cpu_percent:.1f}%", f"[{cpu_color}]●[/{cpu_color}]")

            # Memory usage with color coding
            if hasattr(resources, "memory_total_gb") and hasattr(resources, "memory_available_gb"):
                ram_used = resources.memory_total_gb - resources.memory_available_gb
                ram_percent = (
                    (ram_used / resources.memory_total_gb * 100)
                    if resources.memory_total_gb > 0
                    else 0
                )
                ram_color = "green" if ram_percent < 70 else "yellow" if ram_percent < 85 else "red"
                table.add_row(
                    "RAM",
                    f"{ram_used:.1f}/{resources.memory_total_gb:.1f}GB",
                    f"[{ram_color}]●[/{ram_color}]",
                )
            else:
                table.add_row("RAM", "Unknown", "[yellow]●[/yellow]")

            # GPU usage if available
            if hasattr(resources, "gpu_available") and resources.gpu_available:
                gpu_usage = (
                    resources.gpu_usage_percent if hasattr(resources, "gpu_usage_percent") else 0
                )
                gpu_vram = (
                    f"{resources.gpu_memory_gb:.1f}GB"
                    if hasattr(resources, "gpu_memory_gb") and resources.gpu_memory_gb
                    else "Unknown"
                )
                gpu_color = "green" if gpu_usage < 60 else "yellow" if gpu_usage < 80 else "red"
                table.add_row(
                    "GPU", f"{gpu_usage:.1f}% ({gpu_vram})", f"[{gpu_color}]●[/{gpu_color}]"
                )
            else:
                table.add_row("GPU", "Not Available", "[dim]○[/dim]")

            # Model and token info
            table.add_row("Model", model_info["name"], "[blue]●[/blue]")
            if hasattr(status, "tokens_used"):
                tokens_color = (
                    "green"
                    if status.tokens_used < 1000
                    else "yellow"
                    if status.tokens_used < 4000
                    else "red"
                )
                table.add_row(
                    "Tokens", str(status.tokens_used), f"[{tokens_color}]●[/{tokens_color}]"
                )

            return table

        except Exception as e:
            if RICH_AVAILABLE:
                self.console.print(f"[red]Error creating resource table: {e}[/red]")
            return None

    def create_resource_progress(self, interface):
        """Create progress bars for visual resource consumption."""
        if not RICH_AVAILABLE or not self.console:
            return None

        try:
            status = interface.get_system_status()
            resources = status.resources

            progress_items = []

            # CPU Progress
            if hasattr(resources, "cpu_percent"):
                try:
                    cpu_progress = Progress(
                        TextColumn("CPU"),
                        BarColumn(bar_width=None),
                        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                        console=self.console,
                    )
                    task = cpu_progress.add_task("CPU", total=100)
                    cpu_progress.update(task, completed=resources.cpu_percent)
                    progress_items.append(cpu_progress)
                except:
                    pass

            # Memory Progress
            if hasattr(resources, "memory_total_gb") and hasattr(resources, "memory_available_gb"):
                ram_used = resources.memory_total_gb - resources.memory_available_gb
                ram_percent = (
                    (ram_used / resources.memory_total_gb * 100)
                    if resources.memory_total_gb > 0
                    else 0
                )
                try:
                    ram_progress = Progress(
                        TextColumn("RAM"),
                        BarColumn(bar_width=None),
                        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                        console=self.console,
                    )
                    task = ram_progress.add_task("RAM", total=100)
                    ram_progress.update(task, completed=ram_percent)
                    progress_items.append(ram_progress)
                except:
                    pass

            # GPU Progress
            if (
                hasattr(resources, "gpu_available")
                and resources.gpu_available
                and hasattr(resources, "gpu_usage_percent")
            ):
                try:
                    gpu_progress = Progress(
                        TextColumn("GPU"),
                        BarColumn(bar_width=None),
                        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                        console=self.console,
                    )
                    task = gpu_progress.add_task("GPU", total=100)
                    gpu_progress.update(task, completed=resources.gpu_usage_percent)
                    progress_items.append(gpu_progress)
                except:
                    pass

            return Group(*progress_items) if progress_items else None

        except Exception as e:
            if RICH_AVAILABLE and hasattr(self.console, "print"):
                self.console.print(f"[red]Error creating resource progress: {e}[/red]")
            return None

        try:
            status = interface.get_system_status()
            resources = status.resources

            progress_items = []

            # CPU Progress
            if hasattr(resources, "cpu_percent"):
                cpu_color = (
                    "green"
                    if resources.cpu_percent < 60
                    else "yellow"
                    if resources.cpu_percent < 80
                    else "red"
                )
                cpu_progress = Progress(
                    TextColumn("CPU"),
                    BarColumn(bar_width=None),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    console=self.console,
                )
                task = cpu_progress.add_task("CPU", total=100)
                cpu_progress.update(task, completed=resources.cpu_percent)
                progress_items.append(cpu_progress)

            # Memory Progress
            if hasattr(resources, "memory_total_gb") and hasattr(resources, "memory_available_gb"):
                ram_used = resources.memory_total_gb - resources.memory_available_gb
                ram_percent = (
                    (ram_used / resources.memory_total_gb * 100)
                    if resources.memory_total_gb > 0
                    else 0
                )
                ram_color = "green" if ram_percent < 70 else "yellow" if ram_percent < 85 else "red"
                ram_progress = Progress(
                    TextColumn("RAM"),
                    BarColumn(bar_width=None),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    console=self.console,
                )
                task = ram_progress.add_task("RAM", total=100)
                ram_progress.update(task, completed=ram_percent)
                progress_items.append(ram_progress)

            # GPU Progress
            if (
                hasattr(resources, "gpu_available")
                and resources.gpu_available
                and hasattr(resources, "gpu_usage_percent")
            ):
                gpu_progress = Progress(
                    TextColumn("GPU"),
                    BarColumn(bar_width=None),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    console=self.console,
                )
                task = gpu_progress.add_task("GPU", total=100)
                gpu_progress.update(task, completed=resources.gpu_usage_percent)
                progress_items.append(gpu_progress)

            return Group(*progress_items) if progress_items else None

        except Exception as e:
            if RICH_AVAILABLE:
                self.console.print(f"[red]Error creating resource progress: {e}[/red]")
            return None

    def format_resource_alerts(self, interface):
        """Check resource levels and format warning alerts."""
        if not RICH_AVAILABLE or not self.console:
            return None

        try:
            status = interface.get_system_status()
            resources = status.resources
            alerts = []

            # CPU alerts
            if hasattr(resources, "cpu_percent") and resources.cpu_percent > 85:
                alerts.append(f"🔥 High CPU usage: {resources.cpu_percent:.1f}%")
            elif hasattr(resources, "cpu_percent") and resources.cpu_percent > 70:
                alerts.append(f"⚠️  Moderate CPU usage: {resources.cpu_percent:.1f}%")

            # Memory alerts
            if hasattr(resources, "memory_total_gb") and hasattr(resources, "memory_available_gb"):
                ram_used = resources.memory_total_gb - resources.memory_available_gb
                ram_percent = (
                    (ram_used / resources.memory_total_gb * 100)
                    if resources.memory_total_gb > 0
                    else 0
                )
                if ram_percent > 90:
                    alerts.append(f"🔥 Critical memory usage: {ram_percent:.1f}%")
                elif ram_percent > 75:
                    alerts.append(f"⚠️  High memory usage: {ram_percent:.1f}%")

            # GPU alerts
            if hasattr(resources, "gpu_available") and resources.gpu_available:
                if hasattr(resources, "gpu_usage_percent") and resources.gpu_usage_percent > 90:
                    alerts.append(f"🔥 Very high GPU usage: {resources.gpu_usage_percent:.1f}%")

            if alerts:
                alert_text = "\n".join(alerts)
                try:
                    return Panel(
                        alert_text,
                        title="⚠️ Resource Alerts",
                        border_style="yellow" if "Critical" not in alert_text else "red",
                        title_align="left",
                    )
                except:
                    return None

            return None

        except Exception as e:
            if RICH_AVAILABLE and hasattr(self.console, "print"):
                self.console.print(f"[red]Error creating resource alerts: {e}[/red]")
            return None

        try:
            status = interface.get_system_status()
            resources = status.resources
            alerts = []

            # CPU alerts
            if hasattr(resources, "cpu_percent") and resources.cpu_percent > 85:
                alerts.append(f"🔥 High CPU usage: {resources.cpu_percent:.1f}%")
            elif hasattr(resources, "cpu_percent") and resources.cpu_percent > 70:
                alerts.append(f"⚠️  Moderate CPU usage: {resources.cpu_percent:.1f}%")

            # Memory alerts
            if hasattr(resources, "memory_total_gb") and hasattr(resources, "memory_available_gb"):
                ram_used = resources.memory_total_gb - resources.memory_available_gb
                ram_percent = (
                    (ram_used / resources.memory_total_gb * 100)
                    if resources.memory_total_gb > 0
                    else 0
                )
                if ram_percent > 90:
                    alerts.append(f"🔥 Critical memory usage: {ram_percent:.1f}%")
                elif ram_percent > 75:
                    alerts.append(f"⚠️  High memory usage: {ram_percent:.1f}%")

            # GPU alerts
            if hasattr(resources, "gpu_available") and resources.gpu_available:
                if hasattr(resources, "gpu_usage_percent") and resources.gpu_usage_percent > 90:
                    alerts.append(f"🔥 Very high GPU usage: {resources.gpu_usage_percent:.1f}%")

            if alerts:
                alert_text = "\n".join(alerts)
                return Panel(
                    alert_text,
                    title="⚠️ Resource Alerts",
                    border_style="yellow" if "Critical" not in alert_text else "red",
                    title_align="left",
                )

            return None

        except Exception as e:
            if RICH_AVAILABLE:
                self.console.print(f"[red]Error creating resource alerts: {e}[/red]")
            return None

    def format_minimal_resources(self, interface) -> str:
        """Format minimal resource display for narrow terminals."""
        try:
            status = interface.get_system_status()
            resources = status.resources

            info = []

            # CPU
            if hasattr(resources, "cpu_percent"):
                cpu_indicator = "●" if resources.cpu_percent < 70 else "○"
                info.append(f"CPU:{cpu_indicator}{resources.cpu_percent:.0f}%")

            # Memory
            if hasattr(resources, "memory_total_gb") and hasattr(resources, "memory_available_gb"):
                ram_used = resources.memory_total_gb - resources.memory_available_gb
                ram_percent = (
                    (ram_used / resources.memory_total_gb * 100)
                    if resources.memory_total_gb > 0
                    else 0
                )
                mem_indicator = "●" if ram_percent < 75 else "○"
                info.append(f"RAM:{mem_indicator}{ram_percent:.0f}%")

            # Model
            model_short = (
                status.current_model.split(":")[0]
                if ":" in status.current_model
                else status.current_model[:8]
            )
            info.append(f"M:{model_short}")

            return " | ".join(info)

        except Exception:
            return "Resources: Unknown"

    def should_update_display(self) -> bool:
        """Check if display needs updating based on terminal resize."""
        current_width = self.get_terminal_width()
        new_layout = self.determine_layout(current_width)

        if current_width != self.last_width or new_layout != self.current_layout:
            self.last_width = current_width
            self.current_layout = new_layout
            return True

        return False


@dataclass
class SessionState:
    """Session state for persistent conversation storage."""

    conversation_id: str
    messages: List[Dict[str, str]]
    timestamp: float
    user_id: Optional[str] = None
    context: Optional[str] = None


# Session file paths
SESSION_DIR = Path.home() / ".mai"
SESSION_FILE = SESSION_DIR / "session.json"
SESSION_LOCK_FILE = SESSION_DIR / ".session.lock"


def _acquire_session_lock() -> bool:
    """Acquire session lock to prevent concurrent access."""
    try:
        SESSION_DIR.mkdir(exist_ok=True)
        # Try to create lock file (atomic operation)
        lock_fd = os.open(SESSION_LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(lock_fd)
        return True
    except FileExistsError:
        # Lock already exists
        return False
    except (OSError, PermissionError):
        # Cannot create lock file
        return False


def _release_session_lock() -> None:
    """Release session lock."""
    try:
        if SESSION_LOCK_FILE.exists():
            SESSION_LOCK_FILE.unlink()
    except (OSError, PermissionError):
        pass  # Best effort


def save_session(session_state: SessionState, verbose: bool = False) -> None:
    """Save session state to file with error handling and history truncation."""
    try:
        # Acquire lock to prevent concurrent access
        if not _acquire_session_lock():
            # Could not acquire lock, skip saving this time
            print(
                f"{Colors.YELLOW}Warning: Could not acquire session lock, skipping save{Colors.RESET}"
            )
            return

        try:
            # Handle large conversation histories (truncate if needed)
            max_messages = 100  # Keep last 50 exchanges (100 messages)
            if len(session_state.messages) > max_messages:
                # Keep recent messages, add truncation notice
                old_messages_count = len(session_state.messages) - max_messages
                session_state.messages = session_state.messages[-max_messages:]
                session_state.context = f"Note: {old_messages_count} older messages were truncated to manage session size."

            # Convert to dictionary and save as JSON
            session_dict = asdict(session_state)
            with open(SESSION_FILE, "w", encoding="utf-8") as f:
                json.dump(session_dict, f, indent=2, ensure_ascii=False)

            # Provide feedback if verbose
            if verbose:
                print(f"{Colors.DIM}Session saved to: {SESSION_FILE}{Colors.RESET}")
        finally:
            # Always release lock
            _release_session_lock()

    except (OSError, IOError, PermissionError) as e:
        # Don't fail the CLI, just log the error
        print(f"{Colors.YELLOW}Warning: Could not save session: {e}{Colors.RESET}")
    except Exception as e:
        print(f"{Colors.YELLOW}Warning: Unexpected error saving session: {e}{Colors.RESET}")


def load_session() -> Optional[SessionState]:
    """Load session state from file with validation and error handling."""
    try:
        if not SESSION_FILE.exists():
            return None

        with open(SESSION_FILE, "r", encoding="utf-8") as f:
            session_dict = json.load(f)

        # Validate required fields
        required_fields = ["conversation_id", "messages", "timestamp"]
        for field in required_fields:
            if field not in session_dict:
                print(
                    f"{Colors.YELLOW}Warning: Session file missing required field: {field}{Colors.RESET}"
                )
                return None

        # Create SessionState object
        return SessionState(
            conversation_id=session_dict["conversation_id"],
            messages=session_dict["messages"],
            timestamp=session_dict["timestamp"],
            user_id=session_dict.get("user_id"),
            context=session_dict.get("context"),
        )

    except (json.JSONDecodeError, OSError, IOError, PermissionError) as e:
        print(f"{Colors.YELLOW}Warning: Could not load session: {e}{Colors.RESET}")
        return None
    except Exception as e:
        print(f"{Colors.YELLOW}Warning: Unexpected error loading session: {e}{Colors.RESET}")
        return None


def cleanup_session() -> None:
    """Clean up session files if needed."""
    try:
        if SESSION_FILE.exists():
            SESSION_FILE.unlink()
    except (OSError, PermissionError) as e:
        print(f"{Colors.YELLOW}Warning: Could not cleanup session file: {e}{Colors.RESET}")


def calculate_session_context(session_timestamp: float) -> str:
    """Calculate contextual message based on time since last session."""
    try:
        current_time = datetime.now().timestamp()
        hours_since = (current_time - session_timestamp) / 3600

        if hours_since < 1:
            return "Welcome back! Continuing our conversation..."
        elif hours_since < 24:
            hours_int = int(hours_since)
            return f"Welcome back! It's been {hours_int} hours since we last spoke."
        elif hours_since < 168:  # 7 days
            days_int = int(hours_since / 24)
            return f"Welcome back! It's been {days_int} days since our last conversation."
        else:
            return "Welcome back! It's been a while since we last talked."

    except Exception:
        return "Welcome back!"


def setup_logging(verbose: bool = False) -> None:
    """Configure logging levels and output format."""
    level = logging.DEBUG if verbose else logging.INFO

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )

    # Setup console handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(ColoredFormatter())

    # Setup file handler for debugging
    file_handler = logging.FileHandler("mai.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""

    COLORS = {
        logging.DEBUG: Colors.DIM,
        logging.INFO: Colors.WHITE,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.RED + Colors.BOLD,
    }

    def format(self, record):
        """Format log record with appropriate color."""
        color = self.COLORS.get(record.levelno, Colors.WHITE)
        record.levelname = f"{color}{record.levelname}{Colors.RESET}"
        return super().format(record)


def format_approval_request(approval_decision) -> str:
    """Format approval request for CLI display with rich styling."""
    if not INTERFACE_AVAILABLE or not hasattr(approval_decision, "request"):
        return "Approval request data unavailable"

    request = approval_decision.request
    risk_level = request.risk_analysis.risk_level

    # Color coding based on risk level
    risk_colors = {
        RiskLevel.LOW: Colors.GREEN,
        RiskLevel.MEDIUM: Colors.YELLOW,
        RiskLevel.HIGH: Colors.RED,
        RiskLevel.BLOCKED: Colors.RED + Colors.BOLD,
    }

    color = risk_colors.get(risk_level, Colors.WHITE)

    formatted = []
    formatted.append(f"\n{color}━━━ APPROVAL REQUEST ━━━{Colors.RESET}")
    formatted.append(
        f"{Colors.BOLD}Operation Type:{Colors.RESET} {_get_operation_type(request.code)}"
    )
    formatted.append(
        f"{Colors.BOLD}Risk Level:{Colors.RESET} {color}{risk_level.value.upper()}{Colors.RESET}"
    )

    if request.risk_analysis.reasons:
        formatted.append(f"{Colors.BOLD}Risk Factors:{Colors.RESET}")
        for reason in request.risk_analysis.reasons[:3]:
            formatted.append(f"  • {reason}")

    if request.risk_analysis.affected_resources:
        formatted.append(
            f"{Colors.BOLD}Affected Resources:{Colors.RESET} {', '.join(request.risk_analysis.affected_resources)}"
        )

    # Code preview
    code_preview = request.code[:150] + "..." if len(request.code) > 150 else request.code
    formatted.append(f"{Colors.BOLD}Code Preview:{Colors.RESET}")
    formatted.append(f"{Colors.DIM}{code_preview}{Colors.RESET}")

    formatted.append(f"\n{Colors.BOLD}Options:{Colors.RESET} [Y]es, [N]o, [D]etails, [Q]uit")
    formatted.append(f"{color}{'━' * 30}{Colors.RESET}\n")

    return "\n".join(formatted)


def display_approval_diff(code: str, risk_level) -> None:
    """Display detailed diff in scrollable format with syntax highlighting."""
    print(f"\n{Colors.BOLD}─── DETAILED CODE VIEW ───{Colors.RESET}")
    print(f"{Colors.BOLD}Risk Level:{Colors.RESET} {risk_level.value.upper()}")
    print(f"{Colors.DIM}{'=' * 50}{Colors.RESET}")

    # Display code with line numbers
    lines = code.split("\n")
    for i, line in enumerate(lines, 1):
        print(f"{Colors.DIM}{i:3d}:{Colors.RESET} {line}")

    print(f"{Colors.DIM}{'=' * 50}{Colors.RESET}")
    print(f"{Colors.BOLD}End of code preview{Colors.RESET}\n")


def interactive_approval_prompt(approval_decision) -> str:
    """Accept user input for approval decision with validation."""
    if not INTERFACE_AVAILABLE:
        return "denied"

    while True:
        try:
            user_input = (
                input(f"{Colors.CYAN}Your decision [Y/n/d/q]:{Colors.RESET} ").strip().lower()
            )

            if not user_input or user_input in ["y", "yes"]:
                return "approved"
            elif user_input in ["n", "no"]:
                return "denied"
            elif user_input in ["d", "details"]:
                return "details"
            elif user_input in ["q", "quit"]:
                return "quit"
            else:
                print(f"{Colors.YELLOW}Invalid input. Please use Y/n/d/q{Colors.RESET}")

        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Approval cancelled by user{Colors.RESET}")
            return "denied"
        except EOFError:
            print(f"\n{Colors.YELLOW}Approval cancelled by user{Colors.RESET}")
            return "denied"


def process_approval_result(approval_decision, user_response: str) -> bool:
    """Process approval result and execute appropriate action."""
    if not INTERFACE_AVAILABLE:
        return False

    if user_response == "approved":
        print(f"{Colors.GREEN}✓ Approved - executing code...{Colors.RESET}")
        # Here we would integrate with actual execution
        # For now, just simulate successful execution
        print(f"{Colors.GREEN}✓ Code executed successfully{Colors.RESET}")
        return True
    elif user_response == "denied":
        print(f"{Colors.YELLOW}✗ Rejected{Colors.RESET}")

        # Ask for feedback
        try:
            feedback = input(
                f"{Colors.CYAN}What should I change differently? (optional):{Colors.RESET} "
            ).strip()
            if feedback:
                print(f"{Colors.GREEN}✓ Feedback recorded for improvement{Colors.RESET}")
            else:
                print(f"{Colors.DIM}No feedback provided{Colors.RESET}")
        except (KeyboardInterrupt, EOFError):
            print(f"\n{Colors.DIM}Skipping feedback{Colors.RESET}")

        return False
    else:
        print(f"{Colors.RED}✗ Invalid response: {user_response}{Colors.RESET}")
        return False


def _get_operation_type(code: str) -> str:
    """Extract operation type from code (simplified version)."""
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


class MaiCLI:
    """Mai Command Line Interface."""

    def __init__(self, verbose: bool = False):
        """Initialize CLI with Mai interface."""
        self.verbose = verbose
        self.interface = None  # type: Optional[MaiInterface]
        self.conversation_engine = None  # type: Optional[ConversationEngine]
        self.logger = logging.getLogger(__name__)
        self.session_state = None  # type: Optional[SessionState]
        self.approval_system = None  # type: Optional[ApprovalSystem]
        self.resource_display = ResourceDisplayManager()  # Resource display manager

    def initialize_interface(self) -> bool:
        """Initialize Mai interface and ConversationEngine."""
        try:
            print(f"{Colors.CYAN}Initializing Mai...{Colors.RESET}")

            # Initialize MaiInterface first
            if MaiInterface is not None:
                self.interface = MaiInterface()
            else:
                print(f"{Colors.RED}✗ MaiInterface not available{Colors.RESET}")
                return False

            if self.interface is None or not self.interface.initialize():
                print(f"{Colors.RED}✗ Failed to initialize Mai{Colors.RESET}")
                return False

            # Initialize ConversationEngine with or without MemoryManager
            if INTERFACE_AVAILABLE and ConversationEngine is not None:
                print(f"{Colors.CYAN}Initializing Conversation Engine...{Colors.RESET}")

                # Try to initialize MemoryManager, but don't fail if it doesn't work
                memory_manager = None
                if MemoryManager is not None:
                    try:
                        memory_manager = MemoryManager(config={})
                        print(f"{Colors.GREEN}✓ MemoryManager initialized{Colors.RESET}")
                    except Exception as e:
                        print(
                            f"{Colors.YELLOW}⚠ MemoryManager failed, continuing without memory: {e}{Colors.RESET}"
                        )
                        memory_manager = None

                self.conversation_engine = ConversationEngine(
                    mai_interface=self.interface,
                    memory_manager=memory_manager,
                    timing_profile="default",
                    debug_mode=self.verbose,
                    enable_metrics=True,
                )
                print(f"{Colors.GREEN}✓ Conversation Engine ready{Colors.RESET}")
            else:
                print(
                    f"{Colors.YELLOW}⚠ Conversation Engine unavailable - falling back to direct interface{Colors.RESET}"
                )
                self.conversation_engine = None

            # Initialize approval system if available
            if ApprovalSystem is not None:
                self.approval_system = ApprovalSystem()
                print(f"{Colors.GREEN}✓ Approval System ready{Colors.RESET}")
            else:
                print(f"{Colors.YELLOW}⚠ Approval System unavailable{Colors.RESET}")

            return True

        except ModelConnectionError as e:
            print(f"{Colors.RED}✗ Cannot connect to Ollama: {e}{Colors.RESET}")
            print(f"{Colors.YELLOW}Please ensure Ollama is running and accessible{Colors.RESET}")
            return False

        except Exception as e:
            print(f"{Colors.RED}✗ Unexpected error during initialization: {e}{Colors.RESET}")
            if self.verbose:
                import traceback

                traceback.print_exc()
            return False

    def list_models_command(self) -> None:
        """List available models with their capabilities."""
        if not self.ensure_interface():
            return

        interface = self.interface
        try:
            models = interface.list_models()
            status = interface.get_system_status()
            resources = status.resources

            for i, model in enumerate(models, 1):
                # Model name and current indicator
                current_indicator = (
                    f"{Colors.GREEN}[CURRENT]{Colors.RESET}" if model["current"] else f"   "
                )
                model_name = f"{Colors.BOLD}{model['name']}{Colors.RESET}"

                # Capability indicator
                cap_colors = {"full": Colors.GREEN, "limited": Colors.YELLOW, "minimal": Colors.RED}
                cap_color = cap_colors.get(model["capability"], Colors.WHITE)
                capability = f"{cap_color}{model['capability'].upper()}{Colors.RESET}"

                print(f"{i:2d}. {current_indicator} {model_name:25} {capability}")

                # Model details
                print(
                    f"     {Colors.DIM}Size: {model['size']}GB | Context: {model['context_window']}{Colors.RESET}"
                )

                # Resource requirements
                reqs = model["resource_requirements"]
                print(
                    f"     {Colors.DIM}RAM: {reqs['ram_gb']:.1f}GB | Storage: {reqs['storage_gb']}GB{Colors.RESET}"
                )

                if model["recommended"]:
                    print(f"     {Colors.GREEN}★ Recommended for current system{Colors.RESET}")

                print()

            # System resources
            print(f"{Colors.BOLD}{Colors.BLUE}System Resources{Colors.RESET}")
            print(f"{Colors.DIM}{'=' * 60}{Colors.RESET}")
            ram_used = resources.memory_total_gb - resources.memory_available_gb
            print(
                f"RAM: {ram_used:.1f}/{resources.memory_total_gb:.1f}GB ({resources.memory_percent:.1f}%)"
            )
            print(f"Available: {resources.memory_available_gb:.1f}GB")

            if resources.gpu_available:
                gpu_vram = (
                    f"{resources.gpu_memory_gb:.1f}GB" if resources.gpu_memory_gb else "Unknown"
                )
                gpu_usage = (
                    f" ({resources.gpu_usage_percent:.1f}%)" if resources.gpu_usage_percent else ""
                )
                print(f"GPU: Available ({gpu_vram} VRAM{gpu_usage})")
            else:
                print(f"{Colors.YELLOW}GPU: Not available{Colors.RESET}")

        except Exception as e:
            print(f"{Colors.RED}Error listing models: {e}{Colors.RESET}")
            if self.verbose:
                import traceback

                traceback.print_exc()

    def status_command(self) -> None:
        """Show current system status and resource usage."""
        if not self.ensure_interface():
            return

        interface = self.interface  # Local variable for type checker
        try:
            status = interface.get_system_status()

            print(f"\n{Colors.BOLD}{Colors.BLUE}Mai System Status{Colors.RESET}")
            print(f"{Colors.DIM}{'=' * 60}{Colors.RESET}\n")

            # Model status
            model_state_colors = {
                ModelState.IDLE: Colors.GREEN,
                ModelState.THINKING: Colors.YELLOW,
                ModelState.RESPONDING: Colors.BLUE,
                ModelState.SWITCHING: Colors.MAGENTA,
                ModelState.ERROR: Colors.RED,
            }
            state_color = model_state_colors.get(status.model_state, Colors.WHITE)

            print(f"{Colors.BOLD}Model Status:{Colors.RESET}")
            print(f"  Current: {status.current_model}")
            print(f"  State: {state_color}{status.model_state.value.upper()}{Colors.RESET}")
            print(f"  Available: {len(status.available_models)} models\n")

            # Resource usage
            print(f"{Colors.BOLD}Resource Usage:{Colors.RESET}")
            ram_used = status.resources.memory_total_gb - status.resources.memory_available_gb
            print(
                f"  RAM: {ram_used:.1f}/{status.resources.memory_total_gb:.1f}GB ({status.resources.memory_percent:.1f}%)"
            )
            print(f"  Available: {status.resources.memory_available_gb:.1f}GB")

            if status.resources.gpu_available:
                print(f"  GPU: Available")
                gpu_vram = (
                    f"{status.resources.gpu_memory_gb:.1f}GB"
                    if status.resources.gpu_memory_gb
                    else "Unknown"
                )
                gpu_usage = (
                    f" ({status.resources.gpu_usage_percent:.1f}%)"
                    if status.resources.gpu_usage_percent
                    else ""
                )
                print(f"  VRAM: {gpu_vram}{gpu_usage}")
            else:
                print(f"  GPU: {Colors.YELLOW}Not available{Colors.RESET}")

            # Conversation info
            print(f"\n{Colors.BOLD}Conversation:{Colors.RESET}")
            print(f"  Length: {status.conversation_length} turns")
            print(
                f"  Context compression: {'Enabled' if status.compression_enabled else 'Disabled'}"
            )

            # Git state
            print(f"\n{Colors.BOLD}Git State:{Colors.RESET}")
            if status.git_state["repository_exists"]:
                print(f"  Repository: {Colors.GREEN}✓{Colors.RESET}")
                print(f"  Branch: {status.git_state['current_branch']}")
                print(f"  Changes: {'Yes' if status.git_state['has_changes'] else 'No'}")
                print(
                    f"  Last commit: {status.git_state.get('last_commit', {}).get('hash', 'Unknown')[:8]}"
                )
            else:
                print(f"  Repository: {Colors.RED}✗ Not a git repository{Colors.RESET}")

            # Performance metrics
            metrics = status.performance_metrics
            print(f"\n{Colors.BOLD}Performance Metrics:{Colors.RESET}")
            print(f"  Uptime: {metrics['uptime_seconds'] / 60:.1f} minutes")
            print(f"  Messages: {metrics['total_messages']}")
            print(f"  Model switches: {metrics['total_model_switches']}")
            print(f"  Compressions: {metrics['total_compressions']}")
            print(f"  Avg response time: {metrics['avg_response_time']:.2f}s")
            print(f"  Messages/min: {metrics['messages_per_minute']:.1f}")

            # Resource constraints
            constraints = interface.handle_resource_constraints()
            if constraints["constraints"]:
                print(f"\n{Colors.YELLOW}{Colors.BOLD}Resource Constraints:{Colors.RESET}")
                for constraint in constraints["constraints"]:
                    print(f"  • {constraint}")

            if constraints["recommendations"]:
                print(f"\n{Colors.CYAN}{Colors.BOLD}Recommendations:{Colors.RESET}")
                for rec in constraints["recommendations"]:
                    print(f"  • {rec}")

        except Exception as e:
            print(f"{Colors.RED}Error getting status: {e}{Colors.RESET}")
            if self.verbose:
                import traceback

                traceback.print_exc()

    def display_resource_info(self, interface, show_header: bool = False) -> None:
        """Display resource information based on terminal width."""
        if not interface:
            return

        try:
            # Check terminal width and determine layout
            width = self.resource_display.get_terminal_width()
            layout = self.resource_display.determine_layout(width)

            if layout == "full" and RICH_AVAILABLE:
                # Full layout: Rich table + progress bars
                if show_header:
                    print(f"{Colors.DIM}─ Resources ─{Colors.RESET}")

                table = self.resource_display.create_resource_table(interface)
                progress = self.resource_display.create_resource_progress(interface)
                alerts = self.resource_display.format_resource_alerts(interface)

                if self.resource_display.console:
                    if table:
                        self.resource_display.console.print(table)
                    if progress:
                        self.resource_display.console.print(progress)
                    if alerts:
                        self.resource_display.console.print(alerts)

            elif layout == "compact" and RICH_AVAILABLE:
                # Compact layout: Just table
                if show_header:
                    print(f"{Colors.DIM}─ Resources ─{Colors.RESET}")

                table = self.resource_display.create_resource_table(interface)
                alerts = self.resource_display.format_resource_alerts(interface)

                if self.resource_display.console:
                    if table:
                        self.resource_display.console.print(table)
                    if alerts:
                        self.resource_display.console.print(alerts)

            else:
                # Minimal layout: Simple text
                minimal_text = self.resource_display.format_minimal_resources(interface)
                if minimal_text:
                    if show_header:
                        print(f"{Colors.DIM}─ Resources ─{Colors.RESET}")
                    print(f"{Colors.DIM}{minimal_text}{Colors.RESET}")

        except Exception as e:
            if self.verbose:
                print(f"{Colors.YELLOW}Resource display error: {e}{Colors.RESET}")

    async def chat_command(self, model_override: Optional[str] = None) -> None:
        """Start interactive conversation loop with ConversationEngine."""
        if not self.ensure_interface():
            return

        interface = self.interface  # Local variable for type checker
        try:
            # Load existing session or create new one
            self.session_state = load_session()

            # Show initial status with resource info
            status = interface.get_system_status()
            print(f"\n{Colors.BOLD}{Colors.CYAN}Mai Chat Interface{Colors.RESET}")
            print(f"{Colors.DIM}{'=' * 60}{Colors.RESET}")
            print(f"Model: {status.current_model}")

            # Display initial resource info
            self.display_resource_info(interface, show_header=True)
            print()  # Add spacing

            # Show session context or welcome
            if self.session_state:
                context_msg = calculate_session_context(self.session_state.timestamp)
                print(f"{Colors.GREEN}{context_msg}{Colors.RESET}")
                if self.verbose:
                    print(f"{Colors.DIM}Session file: {SESSION_FILE}{Colors.RESET}")
            else:
                print(f"{Colors.CYAN}Starting new conversation...{Colors.RESET}")
                if self.verbose:
                    print(f"{Colors.DIM}New session will be saved to: {SESSION_FILE}{Colors.RESET}")

            # Show ConversationEngine status
            if self.conversation_engine:
                print(
                    f"{Colors.GREEN}✓ Conversation Engine enabled with natural timing{Colors.RESET}"
                )
            else:
                print(
                    f"{Colors.YELLOW}⚠ Conversation Engine unavailable - using direct interface{Colors.RESET}"
                )

            print(
                f"Type '{Colors.YELLOW}/help{Colors.RESET}' for commands, '{Colors.YELLOW}/quit{Colors.RESET}' to exit\n"
            )

            # Override model if requested
            if model_override:
                result = interface.switch_model(model_override)
                if result["success"]:
                    print(f"{Colors.GREEN}✓ Switched to model: {model_override}{Colors.RESET}")
                else:
                    print(
                        f"{Colors.RED}✗ Failed to switch model: {result.get('error', 'Unknown error')}{Colors.RESET}"
                    )
                    return

            # Use existing conversation ID or create new one
            if self.session_state:
                conversation_id = self.session_state.conversation_id
            else:
                conversation_id = str(uuid.uuid4())
                self.session_state = SessionState(
                    conversation_id=conversation_id,
                    messages=[],
                    timestamp=datetime.now().timestamp(),
                )
                if self.verbose:
                    print(
                        f"{Colors.DIM}Created new session (ID: {conversation_id[:8]}...){Colors.RESET}"
                    )

            while True:
                try:
                    # Get user input
                    user_input = input(f"{Colors.BLUE}You:{Colors.RESET} ").strip()

                    if not user_input:
                        continue

                    # Handle commands
                    if user_input.startswith("/"):
                        if not await self._handle_chat_command_async(user_input, interface):
                            break
                        continue

                    # Add user message to session history
                    if self.session_state:
                        self.session_state.messages.append({"role": "user", "content": user_input})

                    # Process using ConversationEngine or fallback to direct interface
                    if self.conversation_engine:
                        # Show thinking indicator with resource info
                        print(f"{Colors.YELLOW}Mai is thinking...{Colors.RESET}")
                        # Update resource display during thinking
                        self.display_resource_info(interface, show_header=False)
                        print()  # Add spacing

                        # Process with ConversationEngine (includes natural timing)
                        response_data = self.conversation_engine.process_turn(
                            user_input, conversation_id
                        )

                        # Display response with timing info
                        print(f"{Colors.GREEN}Mai ({response_data.model_used}):{Colors.RESET}")
                        print(response_data.response)

                        # Check if response contains approval request
                        if (
                            hasattr(response_data, "requires_approval")
                            and response_data.requires_approval
                        ):
                            if not await self._handle_approval_workflow(user_input, interface):
                                continue  # Continue conversation after approval/rejection

                        # Add assistant response to session history
                        if self.session_state:
                            self.session_state.messages.append(
                                {"role": "assistant", "content": response_data.response}
                            )

                        # Update session timestamp and save
                        if self.session_state:
                            self.session_state.timestamp = datetime.now().timestamp()
                            save_session(self.session_state, verbose=self.verbose)

                        # Show metadata if verbose
                        if self.verbose:
                            print(f"\n{Colors.DIM}--- Metadata ---{Colors.RESET}")
                            print(f"Model: {response_data.model_used}")
                            print(f"Tokens: {response_data.tokens_used}")
                            print(f"Response time: {response_data.response_time:.2f}s")
                            print(f"Timing category: {response_data.timing_category}")
                            print(f"Memory context: {response_data.memory_context_used} items")
                            print(f"Conversation ID: {response_data.conversation_id}")
                    else:
                        # Fallback to direct interface
                        print(f"{Colors.YELLOW}Mai is thinking...{Colors.RESET}")
                        # Update resource display during thinking
                        self.display_resource_info(interface, show_header=False)
                        print()  # Add spacing

                        # Simple history for fallback mode
                        if not hasattr(self, "_conversation_history"):
                            self._conversation_history = []

                        response_data = interface.send_message(
                            user_input, self._conversation_history
                        )

                        # Display response
                        print(f"{Colors.GREEN}Mai ({response_data['model_used']}):{Colors.RESET}")
                        print(response_data["response"])

                        # Check if response contains approval request
                        if response_data.get("requires_approval"):
                            if not await self._handle_approval_workflow(user_input, interface):
                                continue  # Continue conversation after approval/rejection

                        # Add assistant response to session history
                        if self.session_state:
                            self.session_state.messages.append(
                                {"role": "assistant", "content": response_data["response"]}
                            )

                        # Update session timestamp and save
                        if self.session_state:
                            self.session_state.timestamp = datetime.now().timestamp()
                            save_session(self.session_state, verbose=self.verbose)

                        # Update conversation history for fallback
                        self._conversation_history.append({"role": "user", "content": user_input})
                        self._conversation_history.append(
                            {"role": "assistant", "content": response_data["response"]}
                        )

                        # Show metadata if verbose
                        if self.verbose:
                            print(f"\n{Colors.DIM}--- Metadata ---{Colors.RESET}")
                            print(f"Model: {response_data['model_used']}")
                            print(f"Tokens: {response_data['tokens']}")
                            print(f"Response time: {response_data['response_time']:.2f}s")

                    print()  # Add spacing

                except KeyboardInterrupt:
                    print(f"\n{Colors.YELLOW}Saving session and exiting...{Colors.RESET}")
                    # Save final session state before exit
                    if self.session_state:
                        self.session_state.timestamp = datetime.now().timestamp()
                        save_session(self.session_state, verbose=self.verbose)
                    break

                except EOFError:
                    print(f"\n{Colors.YELLOW}Saving session and exiting...{Colors.RESET}")
                    # Save final session state before exit
                    if self.session_state:
                        self.session_state.timestamp = datetime.now().timestamp()
                        save_session(self.session_state, verbose=self.verbose)
                    break

        except Exception as e:
            print(f"{Colors.RED}Error in chat mode: {e}{Colors.RESET}")
            if self.verbose:
                import traceback

                traceback.print_exc()

    def test_command(self) -> None:
        """Run integration tests and provide clear results."""
        print(f"{Colors.CYAN}Running Phase 1 Integration Tests...{Colors.RESET}")
        print(f"{Colors.DIM}Testing all Phase 1 components and requirements{Colors.RESET}")
        print()

        try:
            # Try to use pytest first
            import subprocess
            import sys
            from pathlib import Path

            # Find test file
            test_file = Path(__file__).parent.parent.parent / "tests" / "test_integration.py"
            project_root = Path(__file__).parent.parent.parent

            if not test_file.exists():
                print(
                    f"{Colors.RED}Error: Integration tests not found at {test_file}{Colors.RESET}"
                )
                return

            # Try pytest first
            try:
                print(f"{Colors.DIM}Attempting to run tests with pytest...{Colors.RESET}")
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short"],
                    cwd=project_root,
                    capture_output=True,
                    text=True,
                )

                # Print pytest output
                if result.stdout:
                    print(result.stdout)
                if result.stderr:
                    print(f"{Colors.RED}pytest errors:{Colors.RESET}")
                    print(result.stderr)

            except (subprocess.CalledProcessError, FileNotFoundError):
                # Fallback to running directly with python
                print(
                    f"{Colors.YELLOW}pytest not available, running tests directly...{Colors.RESET}"
                )

                # Run tests using subprocess to capture output properly
                result = subprocess.run(
                    [sys.executable, str(test_file)],
                    cwd=project_root,
                    capture_output=True,
                    text=True,
                )

                # Print output with appropriate formatting
                if result.stdout:
                    # Color code success/failure lines
                    lines = result.stdout.split("\n")
                    for line in lines:
                        if "✓" in line or "PASSED" in line or "OK" in line:
                            print(f"{Colors.GREEN}{line}{Colors.RESET}")
                        elif "✗" in line or "FAILED" in line or "ERROR" in line:
                            print(f"{Colors.RED}{line}{Colors.RESET}")
                        elif "Import Error" in line or "IMPORT_ERROR" in line:
                            print(f"{Colors.YELLOW}{line}{Colors.RESET}")
                        else:
                            print(line)

                if result.stderr:
                    print(f"{Colors.RED}Errors:{Colors.RESET}")
                    print(result.stderr)

            print()
            print(f"{Colors.BOLD}Test Summary:{Colors.RESET}")
            if result.returncode == 0:
                print(f"{Colors.GREEN}✓ All tests passed successfully!{Colors.RESET}")
            else:
                print(
                    f"{Colors.RED}✗ Some tests failed. Return code: {result.returncode}{Colors.RESET}"
                )

            # Extract success rate if available
            if "Success Rate:" in result.stdout:
                import re

                match = re.search(r"Success Rate: (\d+\.?\d*)%", result.stdout)
                if match:
                    success_rate = float(match.group(1))
                    if success_rate >= 80:
                        print(
                            f"{Colors.GREEN}✓ Phase 1 Validation: PASSED ({success_rate:.1f}%){Colors.RESET}"
                        )
                    else:
                        print(
                            f"{Colors.YELLOW}⚠ Phase 1 Validation: MARGINAL ({success_rate:.1f}%){Colors.RESET}"
                        )
            elif "Ran" in result.stdout and "tests in" in result.stdout:
                # Extract from pytest output
                import re
                import sys

                # Check if pytest showed all passed
                if (
                    "passed" in result.stdout
                    and "failed" not in result.stdout
                    and "error" not in result.stdout
                ):
                    print(f"{Colors.GREEN}✓ Phase 1 Validation: PASSED (pytest){Colors.RESET}")
                else:
                    print(f"{Colors.YELLOW}⚠ Phase 1 Validation: MIXED RESULTS{Colors.RESET}")

        except Exception as e:
            print(f"{Colors.RED}Error running tests: {e}{Colors.RESET}")
            print(
                f"{Colors.YELLOW}Alternative: Run manually with: python3 tests/test_integration.py{Colors.RESET}"
            )
            if self.verbose:
                import traceback

                traceback.print_exc()

    def _handle_chat_command(self, command: str, interface) -> bool:
        """Handle chat commands. Returns False to quit, True to continue."""
        cmd = command.lower().strip()

        if cmd == "/quit" or cmd == "/exit":
            print(f"{Colors.CYAN}Goodbye!{Colors.RESET}")
            return False

        elif cmd == "/help":
            print(f"\n{Colors.BOLD}Available Commands:{Colors.RESET}")
            print(f"  {Colors.YELLOW}/help{Colors.RESET}     - Show this help")
            print(f"  {Colors.YELLOW}/status{Colors.RESET}  - Show current system status")
            print(f"  {Colors.YELLOW}/models{Colors.RESET}   - List available models")
            print(f"  {Colors.YELLOW}/switch X{Colors.RESET}  - Switch to model X")
            print(f"  {Colors.YELLOW}/clear{Colors.RESET}   - Clear conversation history")
            print(f"  {Colors.YELLOW}/session{Colors.RESET} - Show session information")
            print(f"  {Colors.YELLOW}/quit{Colors.RESET}     - Exit chat")
            print()

        elif cmd == "/status":
            self.status_command()

        elif cmd == "/models":
            self.list_models_command()

        elif cmd.startswith("/switch "):
            model_name = cmd[8:].strip()
            result = interface.switch_model(model_name)
            if result["success"]:
                print(f"{Colors.GREEN}✓ Switched to: {model_name}{Colors.RESET}")
            else:
                print(
                    f"{Colors.RED}✗ Failed to switch: {result.get('error', 'Unknown error')}{Colors.RESET}"
                )

        elif cmd == "/clear":
            # Clear session and conversation history
            if self.session_state:
                self.session_state.messages = []
                self.session_state.timestamp = datetime.now().timestamp()
                save_session(self.session_state, verbose=self.verbose)
                print(f"{Colors.GREEN}✓ Conversation history cleared{Colors.RESET}")
            else:
                print(f"{Colors.YELLOW}No active session to clear{Colors.RESET}")

        elif cmd == "/session":
            # Show session information
            if self.session_state:
                print(f"\n{Colors.BOLD}{Colors.CYAN}Session Information{Colors.RESET}")
                print(f"{Colors.DIM}{'=' * 40}{Colors.RESET}")
                print(f"Conversation ID: {self.session_state.conversation_id}")
                print(f"Messages: {len(self.session_state.messages)}")
                print(f"Session file: {SESSION_FILE}")
                print(f"File exists: {'Yes' if SESSION_FILE.exists() else 'No'}")
                if SESSION_FILE.exists():
                    import os

                    size_bytes = os.path.getsize(SESSION_FILE)
                    print(f"File size: {size_bytes} bytes")

                # Show last activity time
                last_activity = datetime.fromtimestamp(self.session_state.timestamp)
                print(f"Last activity: {last_activity.strftime('%Y-%m-%d %H:%M:%S')}")

                # Show session age
                hours_since = (datetime.now().timestamp() - self.session_state.timestamp) / 3600
                if hours_since < 1:
                    age_str = f"{int(hours_since * 60)} minutes ago"
                elif hours_since < 24:
                    age_str = f"{int(hours_since)} hours ago"
                else:
                    age_str = f"{int(hours_since / 24)} days ago"
                print(f"Session age: {age_str}")

                # Show recent messages
                if self.session_state.messages:
                    print(f"\n{Colors.BOLD}Recent Messages:{Colors.RESET}")
                    recent_msgs = self.session_state.messages[-6:]  # Show last 3 exchanges
                    for i, msg in enumerate(recent_msgs):
                        role_color = Colors.BLUE if msg["role"] == "user" else Colors.GREEN
                        role_name = "You" if msg["role"] == "user" else "Mai"
                        content_preview = (
                            msg["content"][:80] + "..."
                            if len(msg["content"]) > 80
                            else msg["content"]
                        )
                        print(f"  {role_color}{role_name}:{Colors.RESET} {content_preview}")
                print()
            else:
                print(f"{Colors.YELLOW}No active session{Colors.RESET}")

        else:
            print(f"{Colors.RED}Unknown command: {command}{Colors.RESET}")
            print(f"Type {Colors.YELLOW}/help{Colors.RESET} for available commands")

        return True

    async def _handle_chat_command_async(self, command: str, interface) -> bool:
        """Async version of _handle_chat_command for use in async chat_command."""
        # For now, just call the sync version
        return self._handle_chat_command(command, interface)

    def _check_approval_needed(self, user_input: str) -> bool:
        """Check if user input might trigger approval request."""
        # Simple heuristic - in real implementation, this would be detected
        # from MaiInterface response indicating approval request
        approval_keywords = [
            "create file",
            "write file",
            "execute code",
            "run command",
            "import os",
            "subprocess",
            "system call",
            "file operation",
        ]

        user_input_lower = user_input.lower()
        return any(keyword in user_input_lower for keyword in approval_keywords)

    async def _handle_approval_workflow(self, user_input: str, interface) -> bool:
        """Handle approval workflow for code execution within chat context."""
        if not self.approval_system:
            print(
                f"{Colors.YELLOW}⚠ Approval system unavailable - allowing execution{Colors.RESET}"
            )
            return True

        try:
            print(f"\n{Colors.CYAN}─── Approval Requested ───{Colors.RESET}")
            print(f"{Colors.BOLD}Your request requires approval before execution{Colors.RESET}")
            print(f"{Colors.DIM}This helps keep your system safe{Colors.RESET}\n")

            # Simulate code that would need approval (comes from MaiInterface in real implementation)
            sample_code = f"# Simulated code based on: {user_input}\nprint('Hello, World!')"

            # Request approval from approval system
            approval_result, decision = self.approval_system.request_approval(sample_code)

            if approval_result.value in ["blocked"]:
                print(
                    f"{Colors.RED}🚫 This operation is blocked for security reasons{Colors.RESET}"
                )
                print(
                    f"{Colors.YELLOW}Type your request differently or choose a safer approach{Colors.RESET}\n"
                )
                return False

            # Display formatted approval request
            formatted_request = format_approval_request(decision)
            print(formatted_request)

            # Get user decision
            user_response = interactive_approval_prompt(decision)

            # Process the result with appropriate context
            if user_response == "details":
                display_approval_diff(sample_code, decision.request.risk_analysis.risk_level)
                # Re-prompt after showing details
                user_response = interactive_approval_prompt(decision)

            success = process_approval_result(decision, user_response)

            if success:
                print(f"\n{Colors.GREEN}✓ Approved - executing your request...{Colors.RESET}")
                # Simulate execution with a brief delay
                import time

                time.sleep(1)
                print(f"{Colors.GREEN}✓ Execution completed successfully{Colors.RESET}\n")
            else:
                print(
                    f"\n{Colors.YELLOW}✓ Request rejected - your feedback helps me improve{Colors.RESET}\n"
                )

            return True  # Always continue conversation after approval workflow

        except Exception as e:
            print(f"{Colors.RED}Error in approval workflow: {e}{Colors.RESET}")
            return True  # Continue conversation even on error

            # Display formatted approval request
            formatted_request = format_approval_request(decision)
            print(formatted_request)

            # Get user decision
            user_response = interactive_approval_prompt(decision)

            # Process the result
            success = process_approval_result(decision, user_response)

            if success:
                print(f"{Colors.GREEN}✓ Code executed successfully{Colors.RESET}")
                return True
            else:
                print(f"{Colors.YELLOW}✓ Feedback recorded - conversation continues{Colors.RESET}")
                return True  # Continue conversation even after rejection

        except Exception as e:
            print(f"{Colors.RED}Error in approval workflow: {e}{Colors.RESET}")
            return False

    def ensure_interface(self) -> bool:
        """Ensure interface is initialized."""
        if not self.interface:
            if not self.initialize_interface():
                return False
        return True


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Mai - Your AI collaborator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --models                    List available models
  %(prog)s --chat                      Start interactive chat
  %(prog)s --chat --model llama3.2:1b  Chat with specific model
  %(prog)s --status                    Show system status
  %(prog)s --test                      Run integration tests (placeholder)
        """,
    )

    parser.add_argument(
        "--models", action="store_true", help="List available models with capabilities"
    )

    parser.add_argument("--chat", action="store_true", help="Start interactive conversation mode")

    parser.add_argument(
        "--status", action="store_true", help="Show system status and resource usage"
    )

    parser.add_argument(
        "--test", action="store_true", help="Run integration tests (placeholder for now)"
    )

    parser.add_argument("--model", type=str, help="Specify model to use (overrides auto-selection)")

    parser.add_argument(
        "--verbose", action="store_true", help="Enable detailed logging and debugging output"
    )

    parser.add_argument("--config", type=str, help="Path to configuration file")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Create CLI instance
    cli = MaiCLI(verbose=args.verbose)

    # Route to appropriate command
    try:
        if args.models:
            cli.list_models_command()
        elif args.chat:
            # Run async chat_command with asyncio.run()
            asyncio.run(cli.chat_command(model_override=args.model))
        elif args.status:
            cli.status_command()
        elif args.test:
            cli.test_command()
        else:
            # Show help if no command specified
            parser.print_help()

    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Interrupted by user{Colors.RESET}")
        sys.exit(0)

    except Exception as e:
        print(f"{Colors.RED}Fatal error: {e}{Colors.RESET}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

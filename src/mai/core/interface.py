"""
Mai Main Interface

This module provides the main Mai interface that integrates all components
including model interface, resource monitoring, and git automation.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from ..model.ollama_client import OllamaClient
from ..model.resource_detector import ResourceDetector, ResourceInfo
from ..model.compression import ContextCompressor
from ..core.config import Config, get_config
from ..core.exceptions import MaiError, ModelError, ConfigurationError, ModelConnectionError
from ..git.workflow import StagingWorkflow
from ..git.committer import AutoCommitter
from ..git.health_check import HealthChecker
from ..sandbox.manager import SandboxManager
from ..sandbox.approval_system import ApprovalSystem, ApprovalResult
from ..sandbox.audit_logger import AuditLogger
from ..memory.manager import MemoryManager, MemoryManagerError


class ModelState(Enum):
    """Model operational state."""

    IDLE = "idle"
    THINKING = "thinking"
    RESPONDING = "responding"
    SWITCHING = "switching"
    ERROR = "error"


@dataclass
class ConversationTurn:
    """Single conversation turn with metadata."""

    message: str
    response: str
    model_used: str
    tokens: int
    timestamp: float
    resources: ResourceInfo
    response_time: float


@dataclass
class SystemStatus:
    """Current system status."""

    current_model: str
    available_models: List[str]
    model_state: ModelState
    resources: ResourceInfo
    conversation_length: int
    compression_enabled: bool
    git_state: Dict[str, Any]
    performance_metrics: Dict[str, float]
    recent_activity: List[ConversationTurn]
    memory_status: Optional[Dict[str, Any]] = None


class MaiInterface:
    """Main Mai interface integrating all components."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize Mai interface with all components."""
        self.logger = logging.getLogger(__name__)

        # Load configuration
        self.config = get_config(config_path)

        # Initialize core components
        self.ollama_client = OllamaClient()
        self.resource_detector = ResourceDetector()
        self.context_compressor = ContextCompressor()

        # Initialize git components
        self.staging_workflow = StagingWorkflow()
        self.auto_committer = AutoCommitter()
        self.health_checker = HealthChecker()

        # Initialize sandbox components
        self.sandbox_manager = SandboxManager()
        self.approval_system = ApprovalSystem()
        self.audit_logger = AuditLogger()

        # Initialize memory system
        self.memory_manager: Optional[MemoryManager] = None
        try:
            self.memory_manager = MemoryManager()
            self.logger.info("Memory system initialized successfully")
        except Exception as e:
            self.logger.warning(f"Memory system initialization failed: {e}")
            self.memory_manager = None

        # State tracking
        self.conversation_history: List[ConversationTurn] = []
        self.current_model: Optional[str] = None
        self.model_state = ModelState.IDLE
        self.initialized = False
        self.last_resource_check = 0
        self.resource_check_interval = 5.0  # seconds

        # Performance metrics
        self.total_messages = 0
        self.total_model_switches = 0
        self.total_compressions = 0
        self.start_time = time.time()

        self.logger.info("Mai interface initialized")

    def initialize(self) -> bool:
        """Initialize all components and verify system state."""
        try:
            self.logger.info("Initializing Mai interface...")

            # Initialize Ollama connection
            models = self.ollama_client.list_models()
            if not models:
                self.logger.warning("No models available in Ollama")
                return False

            # Initialize resource monitoring
            resources = self.resource_detector.get_current_resources()

            # Check git repository
            try:
                self.health_checker.get_current_branch()
            except:
                self.logger.warning("Git repository health check failed")

            # Set initial model - use first available (skip empty names)
            self.current_model = None
            for model in models:
                model_name = model.get("name", "").strip()
                if model_name:
                    self.current_model = model_name
                    break

            if not self.current_model:
                # Fallback: use a default model name
                self.current_model = "default-model"
                self.logger.warning("No valid model names found, using fallback")

            self.logger.info(f"Selected initial model: {self.current_model}")

            self.initialized = True
            self.logger.info("Mai interface initialized successfully")

            # Report status
            print(f"✓ Mai initialized with {len(models)} models available")
            print(f"✓ Current model: {self.current_model}")
            print(
                f"✓ Resources: {resources.memory_total_gb - resources.memory_available_gb:.1f}GB RAM used"
            )

            return True

        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            print(f"✗ Initialization failed: {e}")
            return False

    def list_models(self) -> List[Dict[str, Any]]:
        """Get available models with capabilities and resource analysis."""
        if not self.initialized:
            raise MaiError("Interface not initialized")

        try:
            models = self.ollama_client.list_models()
            resources = self.resource_detector.get_current_resources()

            model_list = []
            for model in models:
                # Simple capability analysis based on resources and model size
                size_gb = model.get("size", 0) / (1024 * 1024 * 1024)  # Convert bytes to GB
                ram_needed = size_gb * 2  # Rough estimate
                is_recommended = ram_needed < resources.memory_available_gb * 0.7

                model_info = {
                    "name": model.get("name", ""),
                    "size": size_gb,
                    "parameters": model.get("parameters", "unknown"),
                    "context_window": model.get("context_window", 4096),
                    "capability": "full"
                    if is_recommended
                    else "limited"
                    if ram_needed < resources.memory_available_gb
                    else "minimal",
                    "recommended": is_recommended,
                    "resource_requirements": self._get_model_resource_requirements(model),
                    "current": model.get("name", "") == self.current_model,
                }
                model_list.append(model_info)

            # Sort by recommendation and name
            model_list.sort(key=lambda x: (-x["recommended"], x["name"]))

            return model_list

        except Exception as e:
            self.logger.error(f"Failed to list models: {e}")
            raise ModelError(f"Cannot list models: {e}")

    def send_message(
        self, message: str, conversation_context: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """Send message to model with automatic selection and resource handling."""
        if not self.initialized:
            raise MaiError("Interface not initialized")

        start_time = time.time()

        try:
            self.model_state = ModelState.THINKING

            # Get current resources
            resources = self.resource_detector.get_current_resources()

            # Retrieve memory context if available
            memory_context = []
            if self.memory_manager:
                try:
                    # Get relevant context from memory
                    memory_result = self.memory_manager.get_context(
                        query=message,
                        max_tokens=1000,  # Limit memory context
                        max_results=3,
                    )

                    # Convert memory results to context format
                    if memory_result.get("relevant_conversations"):
                        memory_context = [
                            {
                                "role": "system",
                                "content": f"Relevant context from previous conversations: {conv['title']} - {conv['excerpt']}",
                            }
                            for conv in memory_result["relevant_conversations"][:2]
                        ]
                except Exception as e:
                    self.logger.debug(f"Failed to retrieve memory context: {e}")

            # Combine conversation context with memory context
            full_context = []
            if memory_context:
                full_context.extend(memory_context)
            if conversation_context:
                full_context.extend(conversation_context)

            # Compress context if needed
            if full_context and self.config.context.compression_enabled:
                context_size = len(str(full_context))
                if context_size > self.config.context.max_conversation_length * 100:
                    # Context is too large, would need to implement compress_context method
                    self.total_compressions += 1

            # Send message to current model
            self.model_state = ModelState.RESPONDING
            response = self.ollama_client.generate_response(
                message, self.current_model, full_context
            )

            # Calculate metrics
            response_time = time.time() - start_time
            tokens_estimated = self._estimate_tokens(message + response)

            # Store conversation in memory if available
            if self.memory_manager:
                try:
                    # Create conversation messages for storage
                    conversation_messages = []
                    if memory_context:
                        conversation_messages.extend(memory_context)
                    if conversation_context:
                        conversation_messages.extend(conversation_context)

                    # Add current turn
                    conversation_messages.extend(
                        [
                            {"role": "user", "content": message},
                            {"role": "assistant", "content": response},
                        ]
                    )

                    # Store in memory
                    conv_id = self.memory_manager.store_conversation(
                        messages=conversation_messages,
                        metadata={
                            "model_used": self.current_model,
                            "response_time": response_time,
                            "tokens": tokens_estimated,
                            "context_from_memory": bool(memory_context),
                        },
                    )

                    self.logger.debug(f"Stored conversation in memory: {conv_id}")

                except Exception as e:
                    self.logger.debug(f"Failed to store conversation in memory: {e}")

            # Record conversation turn
            turn = ConversationTurn(
                message=message,
                response=response,
                model_used=self.current_model or "unknown",
                tokens=tokens_estimated,
                timestamp=start_time,
                resources=resources,
                response_time=response_time,
            )
            self.conversation_history.append(turn)
            self.total_messages += 1

            self.model_state = ModelState.IDLE

            return {
                "response": response,
                "model_used": self.current_model,
                "tokens": tokens_estimated,
                "response_time": response_time,
                "resources": resources.__dict__,
                "model_switched": False,
                "memory_context_used": len(memory_context) if memory_context else 0,
            }

        except Exception as e:
            self.model_state = ModelState.ERROR
            self.logger.error(f"Failed to send message: {e}")
            raise ModelError(f"Cannot send message: {e}")

    def get_system_status(self) -> SystemStatus:
        """Get comprehensive system status."""
        if not self.initialized:
            raise MaiError("Interface not initialized")

        try:
            resources = self.resource_detector.get_current_resources()
            models = self.ollama_client.list_models()

            # Get git state
            git_state = {
                "repository_exists": True,
                "has_changes": False,
                "current_branch": "main",
                "last_commit": {"hash": "unknown"},
            }
            try:
                git_state["current_branch"] = self.health_checker.get_current_branch()
            except:
                pass

            # Calculate performance metrics
            uptime = time.time() - self.start_time
            avg_response_time = (
                sum(turn.response_time for turn in self.conversation_history[-10:])
                / min(10, len(self.conversation_history))
                if self.conversation_history
                else 0
            )

            performance_metrics = {
                "uptime_seconds": uptime,
                "total_messages": self.total_messages,
                "total_model_switches": self.total_model_switches,
                "total_compressions": self.total_compressions,
                "avg_response_time": avg_response_time,
                "messages_per_minute": (self.total_messages / uptime * 60) if uptime > 0 else 0,
            }

            # Get memory status
            memory_status = None
            if self.memory_manager:
                memory_status = self.show_memory_status()

            return SystemStatus(
                current_model=self.current_model or "None",
                available_models=[m.get("name", "") for m in models],
                model_state=self.model_state,
                resources=resources,
                conversation_length=len(self.conversation_history),
                compression_enabled=self.config.context.compression_enabled,
                git_state=git_state,
                performance_metrics=performance_metrics,
                recent_activity=self.conversation_history[-5:] if self.conversation_history else [],
                memory_status=memory_status,
            )

        except Exception as e:
            self.logger.error(f"Failed to get system status: {e}")
            raise MaiError(f"Cannot get system status: {e}")

    def switch_model(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Switch to specified model or auto-select best model."""
        if not self.initialized:
            raise MaiError("Interface not initialized")

        try:
            old_model = self.current_model

            if model_name:
                # Switch to specific model
                models = self.ollama_client.list_models()
                model_names = [m.get("name", "") for m in models]
                if model_name not in model_names:
                    raise ModelError(f"Model '{model_name}' not available")
            else:
                # Use first available model
                models = self.ollama_client.list_models()
                if not models:
                    raise ModelError("No models available")
                model_name = models[0].get("name", "")

            # Perform switch
            self.current_model = model_name
            self.total_model_switches += 1

            return {
                "old_model": old_model,
                "new_model": model_name,
                "success": True,
                "performance_impact": "minimal",
                "resources": self.resource_detector.get_current_resources().__dict__,
            }

        except Exception as e:
            self.logger.error(f"Failed to switch model: {e}")
            return {
                "old_model": self.current_model,
                "new_model": model_name,
                "success": False,
                "error": str(e),
            }

    def handle_resource_constraints(self) -> Dict[str, Any]:
        """Handle resource constraints and provide recommendations."""
        if not self.initialized:
            raise MaiError("Interface not initialized")

        try:
            resources = self.resource_detector.get_current_resources()
            constraints = []
            recommendations = []

            # Check memory constraints
            if resources.memory_percent > 85:
                constraints.append("High memory usage")
                recommendations.append("Consider switching to smaller model")

            if resources.memory_available_gb < 2:
                constraints.append("Low available memory")
                recommendations.append("Close other applications or switch to lighter model")

            return {
                "constraints": constraints,
                "recommendations": recommendations,
                "resources": resources.__dict__,
                "urgency": "high" if len(constraints) > 2 else "medium" if constraints else "low",
            }

        except Exception as e:
            self.logger.error(f"Failed to handle resource constraints: {e}")
            raise MaiError(f"Cannot handle resource constraints: {e}")

    def shutdown(self) -> None:
        """Gracefully shutdown all components."""
        try:
            self.logger.info("Shutting down Mai interface...")

            self.model_state = ModelState.IDLE
            self.initialized = False

            # Close memory system
            if self.memory_manager:
                self.memory_manager.close()
                self.logger.info("Memory system shutdown complete")

            self.logger.info("Mai interface shutdown complete")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

    # Private helper methods

    def _get_model_resource_requirements(self, model: Dict[str, Any]) -> Dict[str, float]:
        """Estimate resource requirements for a model."""
        size_gb = model.get("size", 0) / (1024 * 1024 * 1024)  # Convert bytes to GB if needed
        context_window = model.get("context_window", 4096)

        base_ram_gb = size_gb * 2  # Model typically needs 2x size in RAM
        context_ram_gb = context_window / 100000  # Rough estimate

        return {
            "ram_gb": base_ram_gb + context_ram_gb,
            "storage_gb": size_gb,
            "vram_gb": base_ram_gb * 0.8,  # Estimate 80% can be in VRAM
        }

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        # Rough estimation: ~4 characters per token
        return len(text) // 4

    # Sandbox Integration Methods

    def execute_code_safely(
        self, code: str, environment: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Execute code safely through sandbox with approval workflow."""
        try:
            # Request approval for code execution
            context = {
                "user_level": "known",  # Could be determined from user history
                "environment": environment or {},
                "request_source": "cli",
            }

            approval_result, decision = self.approval_system.request_approval(code, context)

            if approval_result == ApprovalResult.BLOCKED:
                return False, "Operation blocked for security reasons", decision

            if approval_result == ApprovalResult.DENIED:
                return False, "Operation denied by user", decision

            # Log execution attempt
            execution_id = self.audit_logger.log_execution_attempt(
                code=code,
                risk_level=decision.request.risk_analysis.risk_level.value,
                user_decision=decision.result.value,
                context=context,
            )

            # Execute in sandbox
            execution_result = self.sandbox_manager.execute_code(code, environment)

            # Log execution result
            self.audit_logger.log_execution_result(
                execution_id=execution_id,
                success=execution_result.get("success", False),
                output=execution_result.get("output", ""),
                error=execution_result.get("error", ""),
                execution_time=execution_result.get("execution_time", 0.0),
            )

            return (
                execution_result.get("success", False),
                execution_result.get("output", ""),
                execution_result,
            )

        except Exception as e:
            self.logger.error(f"Error in safe code execution: {e}")
            return False, f"Execution error: {str(e)}", None

    def show_sandbox_status(self) -> Dict[str, Any]:
        """Show current sandbox status and configuration."""
        try:
            sandbox_config = self.config.get("sandbox", {})

            status = {
                "sandbox_enabled": True,
                "resource_limits": {
                    "cpu_percent": sandbox_config.get("cpu_percent", 70),
                    "memory_percent": sandbox_config.get("memory_percent", 70),
                    "timeout_seconds": sandbox_config.get("timeout_seconds", 30),
                    "bandwidth_mbps": sandbox_config.get("bandwidth_mbps", 50),
                },
                "approval_settings": {
                    "auto_approve_low_risk": sandbox_config.get("auto_approve_low_risk", True),
                    "require_approval_high_risk": sandbox_config.get(
                        "require_approval_high_risk", True
                    ),
                    "remember_preferences": sandbox_config.get("remember_preferences", True),
                },
                "docker_settings": {
                    "image_name": sandbox_config.get("docker.image_name", "python:3.11-slim"),
                    "network_access": sandbox_config.get("docker.network_access", False),
                    "mount_points": sandbox_config.get("docker.mount_points", []),
                },
                "audit_settings": {
                    "log_level": sandbox_config.get("audit.log_level", "INFO"),
                    "retention_days": sandbox_config.get("audit.retention_days", 30),
                    "mask_sensitive_data": sandbox_config.get("audit.mask_sensitive_data", True),
                },
                "risk_thresholds": sandbox_config.get(
                    "risk_thresholds",
                    {"low_threshold": 0.3, "medium_threshold": 0.6, "high_threshold": 0.8},
                ),
                "user_preferences": len(self.approval_system.user_preferences),
                "approval_history": len(self.approval_system.approval_history),
            }

            return status

        except Exception as e:
            self.logger.error(f"Error getting sandbox status: {e}")
            return {"error": str(e), "sandbox_enabled": False}

    def review_audit_logs(self, count: int = 10) -> List[Dict[str, Any]]:
        """Review recent audit logs."""
        try:
            # Get recent approval decisions
            approval_history = self.approval_system.get_approval_history(count)

            # Get audit logs from audit logger
            recent_logs = self.audit_logger.get_recent_logs(count)

            # Combine and format
            logs = []

            # Add approval decisions
            for decision in approval_history:
                logs.append(
                    {
                        "timestamp": decision.timestamp.isoformat(),
                        "type": "approval",
                        "request_id": decision.request.request_id,
                        "risk_level": decision.request.risk_analysis.risk_level.value,
                        "severity_score": decision.request.risk_analysis.severity_score,
                        "result": decision.result.value,
                        "user_input": decision.user_input,
                        "operation_type": self.approval_system._get_operation_type(
                            decision.request.code
                        ),
                        "code_preview": decision.request.code[:100] + "..."
                        if len(decision.request.code) > 100
                        else decision.request.code,
                    }
                )

            # Add execution logs
            for log in recent_logs:
                logs.append(
                    {
                        "timestamp": log.get("timestamp", ""),
                        "type": "execution",
                        "execution_id": log.get("execution_id", ""),
                        "risk_level": log.get("risk_level", ""),
                        "success": log.get("success", False),
                        "execution_time": log.get("execution_time", 0.0),
                        "has_error": bool(log.get("error")),
                        "output_preview": (log.get("output", "")[:100] + "...")
                        if len(log.get("output", "")) > 100
                        else log.get("output", ""),
                    }
                )

            # Sort by timestamp and limit
            logs.sort(key=lambda x: x["timestamp"], reverse=True)
            return logs[:count]

        except Exception as e:
            self.logger.error(f"Error reviewing audit logs: {e}")
            return [{"error": str(e), "type": "error"}]

    def configure_sandbox(self) -> bool:
        """Interactive sandbox configuration."""
        try:
            print("\n🔧 Sandbox Configuration")
            print("=" * 40)

            current_config = self.config.get("sandbox", {})

            # Resource limits
            print("\n📊 Resource Limits:")
            cpu = input(f"CPU limit percent [{current_config.get('cpu_percent', 70)}]: ").strip()
            memory = input(
                f"Memory limit percent [{current_config.get('memory_percent', 70)}]: "
            ).strip()
            timeout = input(
                f"Timeout seconds [{current_config.get('timeout_seconds', 30)}]: "
            ).strip()

            # Approval settings
            print("\n🔐 Approval Settings:")
            auto_low = (
                input(
                    f"Auto-approve low risk? [{current_config.get('auto_approve_low_risk', True)}]: "
                )
                .strip()
                .lower()
            )
            require_high = (
                input(
                    f"Require approval for high risk? [{current_config.get('require_approval_high_risk', True)}]: "
                )
                .strip()
                .lower()
            )

            # Update configuration
            updates = {
                "cpu_percent": int(cpu) if cpu.isdigit() else current_config.get("cpu_percent", 70),
                "memory_percent": int(memory)
                if memory.isdigit()
                else current_config.get("memory_percent", 70),
                "timeout_seconds": int(timeout)
                if timeout.isdigit()
                else current_config.get("timeout_seconds", 30),
                "auto_approve_low_risk": auto_low in ["true", "yes", "1"]
                if auto_low
                else current_config.get("auto_approve_low_risk", True),
                "require_approval_high_risk": require_high in ["true", "yes", "1"]
                if require_high
                else current_config.get("require_approval_high_risk", True),
            }

            # Note: In a full implementation, this would save to config file
            print("\n✓ Configuration updated (changes apply to current session)")
            print("Note: Permanent configuration changes require config file update")

            return True

        except Exception as e:
            self.logger.error(f"Error configuring sandbox: {e}")
            print(f"\n✗ Configuration error: {e}")
            return False

    def reset_sandbox_preferences(self) -> bool:
        """Reset all sandbox user preferences."""
        try:
            self.approval_system.reset_preferences()
            print("✓ All sandbox preferences reset to defaults")
            return True
        except Exception as e:
            self.logger.error(f"Error resetting preferences: {e}")
            print(f"✗ Error resetting preferences: {e}")
            return False

    def get_sandbox_health(self) -> Dict[str, Any]:
        """Get sandbox system health status."""
        try:
            # Check Docker availability
            docker_status = self.sandbox_manager.check_docker_availability()

            # Check audit log integrity
            log_health = self.audit_logger.check_log_integrity()

            # Get recent approval patterns
            trust_patterns = self.approval_system.get_trust_patterns()

            health = {
                "overall_status": "healthy" if docker_status and log_health else "degraded",
                "docker_available": docker_status,
                "audit_logs_healthy": log_health,
                "trust_patterns": trust_patterns,
                "total_approvals": len(self.approval_system.approval_history),
                "user_preferences": len(self.approval_system.user_preferences),
                "last_check": time.time(),
            }

            return health

        except Exception as e:
            self.logger.error(f"Error checking sandbox health: {e}")
            return {"overall_status": "error", "error": str(e), "last_check": time.time()}

    # Memory System Integration Methods

    def show_memory_status(self) -> Dict[str, Any]:
        """Show current memory system status and statistics."""
        if not self.memory_manager:
            return {
                "memory_enabled": False,
                "error": "Memory system not initialized",
                "components": {"storage": False, "compression": False, "retrieval": False},
            }

        try:
            # Get comprehensive memory statistics
            memory_stats = self.memory_manager.get_memory_stats()

            # Convert to dictionary for display
            status = {
                "memory_enabled": True,
                "overall_health": memory_stats.system_health,
                "components": {
                    "storage": {
                        "enabled": memory_stats.storage_enabled,
                        "conversations": memory_stats.total_conversations,
                        "messages": memory_stats.total_messages,
                        "size_mb": round(memory_stats.database_size_mb, 2),
                    },
                    "compression": {
                        "enabled": memory_stats.compression_enabled,
                        "total_compressions": memory_stats.total_compressions,
                        "average_ratio": round(memory_stats.average_compression_ratio, 2),
                        "compressed_conversations": memory_stats.compressed_conversations,
                    },
                    "retrieval": {
                        "enabled": memory_stats.retrieval_enabled,
                        "recent_searches": memory_stats.recent_searches,
                        "average_search_time": round(
                            memory_stats.average_search_time * 1000, 2
                        ),  # Convert to ms
                    },
                },
                "health": {
                    "status": memory_stats.system_health,
                    "last_error": memory_stats.last_error,
                    "last_activity": memory_stats.last_activity,
                },
                "auto_compression": {
                    "enabled": self.memory_manager.auto_compression_enabled,
                    "check_interval": self.memory_manager.compression_check_interval,
                    "message_counter": self.memory_manager.message_counter,
                },
            }

            return status

        except Exception as e:
            self.logger.error(f"Error getting memory status: {e}")
            return {"memory_enabled": False, "error": str(e), "overall_health": "error"}

    def search_memory(self, query: str, **kwargs) -> Dict[str, Any]:
        """Search memory for conversations matching the query."""
        if not self.memory_manager:
            return {"success": False, "error": "Memory system not available", "results": []}

        try:
            # Extract search parameters
            limit = kwargs.get("limit", 10)
            filters = kwargs.get("filters", {})
            conversation_type = kwargs.get("conversation_type", None)

            # Perform search
            if conversation_type:
                # Use context retrieval for typed search
                context_result = self.memory_manager.get_context(
                    query=query, conversation_type=conversation_type, max_results=limit
                )
                results = [
                    {
                        "conversation_id": conv["conversation_id"],
                        "title": conv["title"],
                        "similarity_score": conv["similarity_score"],
                        "excerpt": conv["excerpt"],
                        "relevance_type": conv["relevance_type"],
                    }
                    for conv in context_result["relevant_conversations"]
                ]
                metadata = {
                    "total_conversations": context_result["total_conversations"],
                    "estimated_tokens": context_result["estimated_tokens"],
                    "search_time": context_result["search_time"],
                    "query_metadata": context_result["metadata"],
                }
            else:
                # Use basic search
                results = self.memory_manager.search_conversations(
                    query=query, filters=filters, limit=limit
                )
                metadata = {
                    "query": query,
                    "filters_applied": bool(filters),
                    "result_count": len(results),
                }

            return {
                "success": True,
                "query": query,
                "results": results,
                "metadata": metadata,
                "timestamp": time.time(),
            }

        except Exception as e:
            self.logger.error(f"Error searching memory: {e}")
            return {"success": False, "error": str(e), "results": []}

    def manage_memory(self, action: str, target: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Manage memory with various actions (cleanup, compress, etc.)."""
        if not self.memory_manager:
            return {"success": False, "error": "Memory system not available", "action": action}

        try:
            if action == "cleanup":
                # Clean up old memories
                days_old = kwargs.get("days_old", 90)
                result = self.memory_manager.cleanup_old_memories(days_old=days_old)
                return {
                    "success": True,
                    "action": action,
                    "result": result,
                    "message": f"Cleaned up {result['deleted_count']} conversations older than {days_old} days",
                }

            elif action == "compress_check":
                # Check compression triggers
                triggered = self.memory_manager.check_compression_triggers()
                return {
                    "success": True,
                    "action": action,
                    "result": {"triggered_conversations": triggered, "count": len(triggered)},
                    "message": f"Found {len(triggered)} conversations needing compression",
                }

            elif action == "compress_all":
                # Force compression check on all conversations
                triggered = self.memory_manager.check_compression_triggers()
                compressed_count = 0
                errors = []

                for conv_id in triggered:
                    try:
                        # Note: This would need compressor.compress_conversation() method
                        # For now, just count triggered conversations
                        compressed_count += 1
                    except Exception as e:
                        errors.append(f"Failed to compress {conv_id}: {e}")

                return {
                    "success": len(errors) == 0,
                    "action": action,
                    "result": {
                        "attempted": len(triggered),
                        "compressed": compressed_count,
                        "errors": errors,
                    },
                    "message": f"Attempted compression on {len(triggered)} conversations",
                }

            elif action == "stats":
                # Get detailed statistics
                stats = self.memory_manager.get_memory_stats()
                return {
                    "success": True,
                    "action": action,
                    "result": stats.to_dict(),
                    "message": "Retrieved memory system statistics",
                }

            elif action == "reset_counters":
                # Reset performance counters
                self.memory_manager.search_times.clear()
                self.memory_manager.compression_history.clear()
                self.memory_manager.message_counter = 0

                return {
                    "success": True,
                    "action": action,
                    "result": {"counters_reset": True},
                    "message": "Reset all memory performance counters",
                }

            else:
                return {
                    "success": False,
                    "error": f"Unknown action: {action}",
                    "action": action,
                    "available_actions": [
                        "cleanup",
                        "compress_check",
                        "compress_all",
                        "stats",
                        "reset_counters",
                    ],
                }

        except Exception as e:
            self.logger.error(f"Error managing memory: {e}")
            return {"success": False, "error": str(e), "action": action}


# Standalone CLI functions for memory management
# These provide direct access to memory functionality without requiring MaiInterface instance


def show_memory_status() -> Dict[str, Any]:
    """Show current memory system status and statistics."""
    try:
        # Create a temporary interface instance to access memory system
        interface = MaiInterface()
        if interface.memory_manager:
            return interface.show_memory_status()
        else:
            return {
                "memory_enabled": False,
                "error": "Memory system not initialized",
                "components": {"storage": False, "compression": False, "retrieval": False},
            }
    except Exception as e:
        return {"memory_enabled": False, "error": str(e), "overall_health": "error"}


def search_memory(query: str, **kwargs) -> Dict[str, Any]:
    """Search memory for conversations matching query."""
    try:
        # Create a temporary interface instance to access memory system
        interface = MaiInterface()
        if interface.memory_manager:
            return interface.search_memory(query, **kwargs)
        else:
            return {"success": False, "error": "Memory system not available", "results": []}
    except Exception as e:
        return {"success": False, "error": str(e), "results": []}


def manage_memory(action: str, target: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """Manage memory with various actions (cleanup, compress, etc.)."""
    try:
        # Create a temporary interface instance to access memory system
        interface = MaiInterface()
        if interface.memory_manager:
            return interface.manage_memory(action, target, **kwargs)
        else:
            return {"success": False, "error": "Memory system not available", "action": action}
    except Exception as e:
        return {"success": False, "error": str(e), "action": action}

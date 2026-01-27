# Architecture

**Analysis Date:** 2026-01-26

## Pattern Overview

**Overall:** Layered modular architecture with clear separation of concerns

**Key Characteristics:**
- Modular layer separation (Model Interface, Memory, Conversation, Interfaces, Safety, Core Personality)
- Local-first, offline-capable design with graceful degradation
- Plugin-like interface system allowing CLI and Discord without tight coupling
- Sandboxed execution environment for self-improvement code
- Bidirectional feedback loops between conversation, memory, and personality

## Layers

**Model Interface (Inference Layer):**
- Purpose: Abstract model inference operations and handle model switching
- Location: `src/models/`
- Contains: Model adapters, resource monitoring, context management
- Depends on: Local Ollama/LMStudio, system resource API
- Used by: Conversation engine, core Mai reasoning

**Memory System (Persistence Layer):**
- Purpose: Store and retrieve conversation history, patterns, learned behaviors
- Location: `src/memory/`
- Contains: SQLite operations, vector search, compression logic, pattern extraction
- Depends on: Local SQLite database, embeddings generation
- Used by: Conversation engine for context retrieval, personality learning

**Conversation Engine (Reasoning Layer):**
- Purpose: Orchestrate multi-turn conversations with context awareness
- Location: `src/conversation/`
- Contains: Turn handling, context window management, clarifying question logic, reasoning transparency
- Depends on: Model Interface, Memory System, Personality System
- Used by: Interface layers (CLI, Discord)

**Personality System (Behavior Layer):**
- Purpose: Enforce core values and enable personality adaptation
- Location: `src/personality/`
- Contains: Core personality rules, learned behavior layers, guardrails, values enforcement
- Depends on: Configuration files (YAML), Memory System for learned patterns
- Used by: Conversation Engine for decision making and refusal logic

**Safety & Execution Sandbox (Security Layer):**
- Purpose: Validate and execute generated code safely with risk assessment
- Location: `src/safety/`
- Contains: Risk analysis, Docker sandbox management, AST validation, audit logging
- Depends on: Docker runtime, code analysis libraries
- Used by: Self-improvement system for generated code execution

**Self-Improvement System (Autonomous Layer):**
- Purpose: Analyze own code, generate improvements, manage review and approval workflow
- Location: `src/selfmod/`
- Contains: Code analysis, improvement generation, review coordination, git integration
- Depends on: Safety layer, second-agent review API, git operations, code parser
- Used by: Core Mai autonomous operation

**Interface Adapters (Presentation Layer):**
- Purpose: Translate between external communication channels and core conversation engine
- Location: `src/interfaces/`
- Contains: CLI handler, Discord bot, message queuing, approval workflow
- Depends on: Conversation Engine, self-improvement system
- Used by: External communication channels (terminal, Discord)

## Data Flow

**Conversation Flow:**

1. User message arrives via interface (CLI or Discord)
2. Message queued if offline, held in memory if online
3. Interface adapter passes to Conversation Engine
4. Conversation Engine queries Memory System for relevant context
5. Context + message passed to Model Interface with system prompt (includes personality)
6. Model generates response
7. Response returned to Conversation Engine
8. Conversation Engine stores turn in Memory System
9. Response sent back through interface to user
10. Memory System may trigger asynchronous compression if history grows

**Self-Improvement Flow:**

1. Self-Improvement System analyzes own code (triggered by timer or explicit request)
2. Generates potential improvements as Python code patches
3. Performs AST validation and basic static analysis
4. Submits for second-agent review with risk classification
5. If LOW risk: auto-approved, sent to Safety layer for execution
6. If MEDIUM risk: user approval required via CLI or Discord reactions
7. If HIGH/BLOCKED risk: blocked, logged, user notified
8. Approved changes executed in Docker sandbox with resource limits
9. Execution results captured, logged, committed to git with clear message
10. Breaking changes require explicit user approval before commit

**State Management:**
- Conversation state: Maintained in Memory System as persisted history
- Model state: Loaded fresh per request, no state persistence between calls
- Personality state: Mix of code-enforced rules and learned behavior layers in Memory
- Resource state: Monitored continuously, triggering model downgrade if limits approached
- Approval state: Tracked in git commits, audit log, and in-memory queue

## Key Abstractions

**ModelAdapter:**
- Purpose: Abstract different model providers (Ollama local models)
- Examples: `src/models/ollama_adapter.py`, `src/models/model_manager.py`
- Pattern: Strategy pattern with resource-aware selection logic

**ContextWindow:**
- Purpose: Manage token budget and conversation history within model limits
- Examples: `src/conversation/context_manager.py`
- Pattern: Intelligent windowing with semantic importance weighting

**MemoryStore:**
- Purpose: Unified interface to conversation history, patterns, and learned behaviors
- Examples: `src/memory/store.py`, `src/memory/vector_search.py`
- Pattern: Repository pattern with multiple index types

**PersonalityRules:**
- Purpose: Encode Mai's core values as evaluable constraints
- Examples: `src/personality/core_rules.py`, `config/personality.yaml`
- Pattern: Rule engine with value-based decision making

**SandboxExecutor:**
- Purpose: Execute generated code safely with resource limits and audit trail
- Examples: `src/safety/executor.py`, `src/safety/risk_analyzer.py`
- Pattern: Facade wrapping Docker API with security checks

**ApprovalWorkflow:**
- Purpose: Coordinate user and agent approval for code changes
- Examples: `src/interfaces/approval_handler.py`, `src/selfmod/reviewer.py`
- Pattern: State machine with async notification coordination

## Entry Points

**CLI Entry:**
- Location: `src/interfaces/cli.py` / `__main__.py`
- Triggers: `python -m mai` or `mai` command
- Responsibilities: Initialize conversation session, handle user input loop, display responses, manage approval prompts

**Discord Entry:**
- Location: `src/interfaces/discord_bot.py`
- Triggers: Discord message events
- Responsibilities: Extract message context, route to conversation engine, format response, handle reactions for approvals

**Self-Improvement Entry:**
- Location: `src/selfmod/scheduler.py`
- Triggers: Timer-based (periodic analysis) or explicit trigger from conversation
- Responsibilities: Analyze code, generate improvements, initiate review workflow

**Core Mai Entry:**
- Location: `src/mai.py` (main class)
- Triggers: System startup
- Responsibilities: Initialize all systems (models, memory, personality), coordinate between layers

## Error Handling

**Strategy:** Graceful degradation with clear user communication

**Patterns:**
- Model unavailable: Fall back to smaller model if available, notify user of reduced capabilities
- Memory retrieval failure: Continue conversation without historical context, log error
- Network error: Queue offline messages, retry on reconnection (Discord only)
- Unsafe code generated: Block execution, log with risk analysis, notify user
- Syntax error in generated code: Reject change, log, generate new proposal

## Cross-Cutting Concerns

**Logging:** Structured logging with severity levels throughout codebase. Use Python `logging` module with JSON formatter for production. Log all: model selections, memory operations, safety decisions, approval workflows, code changes.

**Validation:** Input validation at interface boundaries. AST validation for generated code. Type hints throughout codebase with mypy enforcement.

**Authentication:** None required for local CLI. Discord bot authenticated via token (environment variable). API calls between services use simple function calls (single-process model).

---

*Architecture analysis: 2026-01-26*

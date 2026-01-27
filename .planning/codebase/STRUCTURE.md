# Codebase Structure

**Analysis Date:** 2026-01-26

## Directory Layout

```
mai/
├── src/
│   ├── __main__.py              # CLI entry point
│   ├── mai.py                   # Core Mai class, orchestration
│   ├── models/
│   │   ├── __init__.py
│   │   ├── adapter.py           # Base model adapter interface
│   │   ├── ollama_adapter.py    # Ollama/LMStudio implementation
│   │   ├── model_manager.py     # Model selection and switching logic
│   │   └── resource_monitor.py  # CPU, RAM, GPU tracking
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── store.py             # SQLite conversation store
│   │   ├── vector_search.py     # Semantic similarity search
│   │   ├── compression.py       # History compression and summarization
│   │   └── pattern_extractor.py # Learning and pattern recognition
│   ├── conversation/
│   │   ├── __init__.py
│   │   ├── engine.py            # Main conversation orchestration
│   │   ├── context_manager.py   # Token budget and window management
│   │   ├── turn_handler.py      # Single turn processing
│   │   └── reasoning.py         # Reasoning transparency and clarification
│   ├── personality/
│   │   ├── __init__.py
│   │   ├── core_rules.py        # Unshakeable core values enforcement
│   │   ├── learned_behaviors.py # Personality adaptation from interactions
│   │   ├── guardrails.py        # Safety constraints and refusal logic
│   │   └── config_loader.py     # YAML personality configuration
│   ├── safety/
│   │   ├── __init__.py
│   │   ├── executor.py          # Docker sandbox execution wrapper
│   │   ├── risk_analyzer.py     # Risk classification (LOW/MEDIUM/HIGH/BLOCKED)
│   │   ├── ast_validator.py     # Syntax and import validation
│   │   └── audit_log.py         # Immutable execution history
│   ├── selfmod/
│   │   ├── __init__.py
│   │   ├── analyzer.py          # Code analysis and improvement detection
│   │   ├── generator.py         # Improvement code generation
│   │   ├── scheduler.py         # Periodic and on-demand analysis trigger
│   │   ├── reviewer.py          # Second-agent review coordination
│   │   └── git_manager.py       # Git commit integration
│   ├── interfaces/
│   │   ├── __init__.py
│   │   ├── cli.py               # CLI chat interface
│   │   ├── discord_bot.py       # Discord bot implementation
│   │   ├── message_handler.py   # Shared message processing
│   │   ├── approval_handler.py  # Change approval workflow
│   │   └── offline_queue.py     # Message queueing during disconnection
│   └── utils/
│       ├── __init__.py
│       ├── config.py            # Configuration loading
│       ├── logging.py           # Structured logging setup
│       ├── validators.py        # Input validation helpers
│       └── helpers.py           # Shared utility functions
├── config/
│   ├── personality.yaml         # Core personality configuration
│   ├── models.yaml              # Model definitions and resource limits
│   ├── safety_rules.yaml        # Risk assessment rules
│   └── logging.yaml             # Logging configuration
├── tests/
│   ├── unit/
│   │   ├── test_models.py
│   │   ├── test_memory.py
│   │   ├── test_conversation.py
│   │   ├── test_personality.py
│   │   ├── test_safety.py
│   │   └── test_selfmod.py
│   ├── integration/
│   │   ├── test_conversation_flow.py
│   │   ├── test_selfmod_workflow.py
│   │   └── test_interfaces.py
│   └── fixtures/
│       ├── mock_models.py
│       ├── test_data.py
│       └── sample_conversations.json
├── scripts/
│   ├── setup_ollama.py          # Initial model downloading
│   ├── init_db.py               # Database schema initialization
│   └── verify_environment.py    # Pre-flight checks
├── docker/
│   └── Dockerfile               # Sandbox execution environment
├── .env.example                 # Environment variables template
├── pyproject.toml               # Project metadata and dependencies
├── requirements.txt             # Python dependencies
├── pytest.ini                   # Test configuration
├── Makefile                     # Development commands
└── README.md                    # Project overview
```

## Directory Purposes

**src/:**
- Purpose: All application code
- Contains: Python modules organized by architectural layer
- Key files: `mai.py` (core), `__main__.py` (CLI entry)

**src/models/:**
- Purpose: Model inference abstraction
- Contains: Adapter interfaces, Ollama client, resource monitoring
- Key files: `model_manager.py` (selection logic), `resource_monitor.py` (constraints)

**src/memory/:**
- Purpose: Persistent storage and retrieval
- Contains: SQLite operations, vector search, compression
- Key files: `store.py` (main interface), `vector_search.py` (semantic search)

**src/conversation/:**
- Purpose: Multi-turn conversation orchestration
- Contains: Turn handling, context windowing, reasoning transparency
- Key files: `engine.py` (main coordinator), `context_manager.py` (token budget)

**src/personality/:**
- Purpose: Values enforcement and personality adaptation
- Contains: Core rules, learned behaviors, guardrails
- Key files: `core_rules.py` (unshakeable values), `learned_behaviors.py` (adaptation)

**src/safety/:**
- Purpose: Code execution sandboxing and risk assessment
- Contains: Docker wrapper, AST validation, risk classification, audit logging
- Key files: `executor.py` (sandbox wrapper), `risk_analyzer.py` (classification)

**src/selfmod/:**
- Purpose: Autonomous code improvement and review
- Contains: Code analysis, improvement generation, approval workflow
- Key files: `analyzer.py` (detection), `reviewer.py` (second-agent coordination)

**src/interfaces/:**
- Purpose: External communication adapters
- Contains: CLI handler, Discord bot, approval system
- Key files: `cli.py` (terminal UI), `discord_bot.py` (Discord integration)

**src/utils/:**
- Purpose: Shared utilities and helpers
- Contains: Configuration loading, logging, validation
- Key files: `config.py` (env/file loading), `logging.py` (structured logs)

**config/:**
- Purpose: Non-code configuration files
- Contains: YAML personality, models, safety rules definitions
- Key files: `personality.yaml` (core values), `models.yaml` (resource profiles)

**tests/:**
- Purpose: Test suites organized by type
- Contains: Unit tests (layer isolation), integration tests (flows), fixtures (test data)
- Key files: Each test file mirrors `src/` structure

**scripts/:**
- Purpose: One-off setup and maintenance scripts
- Contains: Database initialization, environment verification
- Key files: `setup_ollama.py` (first-time model setup)

**docker/:**
- Purpose: Container configuration for sandboxed execution
- Contains: Dockerfile for isolation environment
- Key files: `Dockerfile` (build recipe)

## Key File Locations

**Entry Points:**
- `src/__main__.py`: CLI entry, `python -m mai` launches here
- `src/interfaces/discord_bot.py`: Discord bot main loop
- `src/mai.py`: Core Mai class, system initialization

**Configuration:**
- `config/personality.yaml`: Core values, interaction patterns, refusal rules
- `config/models.yaml`: Available models, resource requirements, context windows
- `.env.example`: Required environment variables template

**Core Logic:**
- `src/mai.py`: Main orchestration
- `src/conversation/engine.py`: Conversation turn processing
- `src/selfmod/analyzer.py`: Improvement opportunity detection
- `src/safety/executor.py`: Safe code execution

**Testing:**
- `tests/unit/`: Layer-isolated tests (no dependencies between layers)
- `tests/integration/`: End-to-end flow tests
- `tests/fixtures/`: Mock objects and test data

## Naming Conventions

**Files:**
- Module files: `snake_case.py` (e.g., `model_manager.py`)
- Entry points: `__main__.py` for packages, standalone scripts at package root
- Config files: `snake_case.yaml` (e.g., `personality.yaml`)
- Test files: `test_*.py` (e.g., `test_conversation.py`)

**Directories:**
- Feature areas: `snake_case` (e.g., `src/selfmod/`)
- No abbreviations except `selfmod` (self-modification) which is project standard
- Each layer is a top-level directory under `src/`

**Functions/Classes:**
- Classes: `PascalCase` (e.g., `ModelManager`, `ConversationEngine`)
- Functions: `snake_case` (e.g., `generate_response()`, `validate_code()`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_CONTEXT_TOKENS`)
- Private methods/functions: prefix with `_` (e.g., `_internal_method()`)

**Types:**
- Use type hints throughout: `def process(msg: str) -> str:`
- Complex types in `src/utils/types.py` or local to module

## Where to Add New Code

**New Feature (e.g., new communication interface like Slack):**
- Primary code: `src/interfaces/slack_adapter.py` (new adapter following discord_bot.py pattern)
- Tests: `tests/unit/test_slack_adapter.py` and `tests/integration/test_slack_interface.py`
- Configuration: Add to `src/interfaces/__init__.py` imports and `config/interfaces.yaml` if needed
- Entry hook: Modify `src/mai.py` to initialize new adapter

**New Component/Module (e.g., advanced memory with graph databases):**
- Implementation: `src/memory/graph_store.py` (new module in appropriate layer)
- Interface: Follow existing patterns (e.g., inherit from `src/memory/store.py` base)
- Tests: Corresponding test in `tests/unit/test_memory.py` or new file if complex
- Integration: Modify `src/mai.py` initialization to use new component with feature flag

**Utilities (e.g., new helper function):**
- Shared helpers: `src/utils/helpers.py` (functions) or new file like `src/utils/math_utils.py` if substantial
- Internal helpers: Keep in the module where used (don't over-extract)
- Tests: Add to `tests/unit/test_utils.py`

**Configuration:**
- Static rules: Add to appropriate YAML in `config/`
- Dynamic config: Load in `src/utils/config.py`
- Env-driven: Add to `.env.example` with documentation

## Special Directories

**tests/fixtures/:**
- Purpose: Reusable test data and mock objects
- Generated: No, hand-created
- Committed: Yes, part of repository

**config/:**
- Purpose: Non-code configuration
- Generated: No, hand-maintained
- Committed: Yes, except secrets (use `.env`)

**.env (not committed):**
- Purpose: Local environment overrides and secrets
- Generated: No, copied from `.env.example` and filled locally
- Committed: No (in .gitignore)

**docker/:**
- Purpose: Sandbox environment for safe execution
- Generated: No, hand-maintained
- Committed: Yes

---

*Structure analysis: 2026-01-26*

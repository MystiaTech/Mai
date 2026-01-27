# External Integrations

**Analysis Date:** 2026-01-26

## APIs & External Services

**Model Inference:**
- LMStudio - Local model server for inference and model switching
  - SDK/Client: LMStudio Python API
  - Auth: None (local service, no authentication required)
  - Configuration: model_path env var, endpoint URL

- Ollama - Alternative local model management system
  - SDK/Client: Ollama REST API (HTTP)
  - Auth: None (local service)
  - Purpose: Model loading, switching, inference with resource detection

**Communication & Approvals:**
- Discord - Bot interface for conversation and change approvals
  - SDK/Client: discord.py library
  - Auth: DISCORD_BOT_TOKEN env variable
  - Purpose: Multi-turn conversations, approval reactions (thumbs up/down), status updates

## Data Storage

**Databases:**
- SQLite3 (local file-based)
  - Connection: Local file path, no remote connection
  - Client: Python sqlite3 (stdlib) or SQLAlchemy ORM
  - Purpose: Persistent conversation history, memory compression, learned patterns
  - Location: Local filesystem (.db files)

**File Storage:**
- Local filesystem only - Git-tracked code changes, conversation history backups
- No cloud storage integration in v1

**Caching:**
- In-memory caching for current conversation context
- Redis: Not used in v1 (local-first constraint)
- Model context window management: Token-based cache within model inference

## Authentication & Identity

**Auth Provider:**
- Custom local auth - No external identity provider
- Implementation:
  - Discord user ID as conversation context identifier
  - Optional local password/PIN for CLI access
  - No OAuth/cloud identity providers (offline-first requirement)

## Monitoring & Observability

**Error Tracking:**
- None (local only, no error reporting service)
- Local audit logging to SQLite instead

**Logs:**
- File-based logging to `.logs/` directory
- Format: Structured JSON logs with timestamp, level, context
- Rotation: Size-based or time-based rotation strategy
- No external log aggregation (offline-first)

## CI/CD & Deployment

**Hosting:**
- Local machine only (desktop/laptop with RTX 3060+)
- No cloud hosting in v1

**CI Pipeline:**
- GitHub Actions for Discord webhook on push
  - Workflow: `.github/workflows/discord_sync.yml`
  - Trigger: Push events
  - Action: POST to Discord webhook for notification

**Git Integration:**
- All Mai's self-modifications committed automatically with git
- Local git repo tracking all code changes
- Commit messages include decision context and review results

## Environment Configuration

**Required env vars:**
- `DISCORD_BOT_TOKEN` - Discord bot authentication
- `LMSTUDIO_ENDPOINT` - LMStudio API URL (default: localhost:8000)
- `OLLAMA_ENDPOINT` - Ollama API URL (optional alternative, default: localhost:11434)
- `DISCORD_USER_ID` - User Discord ID for approval requests
- `MEMORY_DB_PATH` - SQLite database file location
- `MODEL_CACHE_DIR` - Directory for model files
- `CPU_CORES_AVAILABLE` - System CPU count for resource management
- `GPU_VRAM_AVAILABLE` - VRAM in GB for model selection
- `SANDBOX_DOCKER_IMAGE` - Docker image ID for code sandbox execution

**Secrets location:**
- `.env` file (Python-dotenv) for local development
- Environment variables for production/runtime
- Git-ignored: `.env` not committed

## Webhooks & Callbacks

**Incoming:**
- Discord message webhooks - Handled by discord.py bot event listeners
- No external webhook endpoints in v1

**Outgoing:**
- Discord webhook for git notifications (configured in GitHub Actions)
- Endpoint: Stored in GitHub secrets as WEBHOOK
- Triggered on: git push events
- Payload: Git commit information (author, message, timestamp)

**Model Callback Handling:**
- LMStudio streaming callbacks for token-by-token responses
- Ollama streaming responses for incremental model output

## Code Execution Sandbox

**Sandbox Environment:**
- Docker container with resource limits
  - SDK: Docker SDK for Python (docker-py)
  - Environment: Isolated Linux container
  - Resource limits: CPU cores, RAM, network restrictions

**Risk Assessment:**
- Multi-level risk evaluation (LOW/MEDIUM/HIGH/BLOCKED)
- AST validation before container execution
- Second-agent review via Claude/OpenCode API

---

*Integration audit: 2026-01-26*

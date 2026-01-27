# Technology Stack

**Analysis Date:** 2026-01-26

## Languages

**Primary:**
- Python 3.x - Core Mai agent codebase, local model inference, self-improvement system

**Secondary:**
- YAML - Configuration files for personality, behavior settings
- JSON - Configuration, metadata, API responses
- SQL - Memory storage and retrieval queries

## Runtime

**Environment:**
- Python (local execution, no remote runtime)
- LMStudio or Ollama - Local model inference server

**Package Manager:**
- pip - Python package management
- Lockfile: requirements.txt or poetry.lock (typical Python approach)

## Frameworks

**Core:**
- No web framework for v1 (CLI/Discord only)

**Model Inference:**
- LMStudio Python SDK - Local model switching and inference
- Ollama API - Alternative local model management per requirements

**Discord Integration:**
- discord.py - Discord bot API client

**CLI:**
- Click or Typer - Command-line interface building

**Testing:**
- pytest - Unit/integration test framework
- pytest-asyncio - Async test support for Discord bot testing

**Build/Dev:**
- Git - Version control for Mai's own code changes
- Docker - Sandbox execution environment for safety

## Key Dependencies

**Critical:**
- LMStudio Python Client - Model loading, switching, inference with token management
- discord.py - Discord bot functionality for approval workflows
- SQLite3 - Lightweight persistent storage (Python stdlib)
- Docker SDK for Python - Sandbox execution management

**Infrastructure:**
- requests - HTTP client for Discord API fallback and Ollama API communication
- PyYAML - Personality configuration parsing
- pydantic - Data validation for internal structures
- python-dotenv - Environment variable management for secrets
- GitPython - Programmatic git operations for committing self-improvements

## Configuration

**Environment:**
- .env file - Discord bot token, model paths, resource thresholds
- environment variables - Runtime configuration loaded at startup
- personality.yaml - Core personality values and learned behavior layers
- config.json - Resource limits, model preferences, memory settings

**Build:**
- setup.py or pyproject.toml - Package metadata and dependency declaration
- Dockerfile - Sandbox execution environment specification
- .dockerignore - Docker build optimization

## Platform Requirements

**Development:**
- Python 3.8+ (for type hints and async/await)
- Git (for version control and self-modification tracking)
- Docker (for sandbox execution environment)
- LMStudio or Ollama running locally (for model inference)

**Production (Runtime):**
- RTX 3060 GPU minimum (per project constraints)
- 16GB+ RAM (for model loading and context management)
- Linux/macOS/Windows with Python 3.8+
- Docker daemon (for sandboxed code execution)
- Local LMStudio/Ollama instance (no cloud models)

---

*Stack analysis: 2026-01-26*

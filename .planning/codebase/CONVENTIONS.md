# Coding Conventions

**Analysis Date:** 2026-01-26

## Status

**Note:** This codebase is in planning phase. No source code has been written yet. These conventions are **prescriptive** for the Mai project and should be applied to all code from the first commit forward.

## Naming Patterns

**Files:**
- Python modules: `lowercase_with_underscores.py` (PEP 8)
- Configuration files: `config.yaml`, `.env.example`
- Test files: `test_module_name.py` (co-located with source)
- Example: `src/memory/storage.py`, `src/memory/test_storage.py`

**Functions:**
- Use `snake_case` for all function names (PEP 8)
- Private functions: Prefix with single underscore `_private_function()`
- Async functions: Use `async def async_operation()` naming
- Example: `def get_conversation_history()`, `async def stream_response()`

**Variables:**
- Use `snake_case` for all variable names
- Constants: `UPPERCASE_WITH_UNDERSCORES`
- Private module variables: Prefix with `_`
- Example: `conversation_history`, `MAX_CONTEXT_TOKENS`, `_internal_cache`

**Types:**
- Classes: `PascalCase`
- Enums: `PascalCase` (inherit from `Enum`)
- TypedDict: `PascalCase` with `Dict` suffix
- Example: `class ConversationManager`, `class ErrorLevel(Enum)`, `class MemoryConfigDict(TypedDict)`

**Directories:**
- Core modules: `src/[module_name]/` (lowercase, plural when appropriate)
- Example: `src/models/`, `src/memory/`, `src/safety/`, `src/interfaces/`

## Code Style

**Formatting:**
- Tool: **Ruff** (formatter and linter)
- Line length: 88 characters (Ruff default)
- Quote style: Double quotes (`"string"`)
- Indentation: 4 spaces (no tabs)

**Linting:**
- Tool: **Ruff**
- Configuration enforced via `.ruff.toml` (when created)
- All imports must pass ruff checks
- No unused imports allowed
- Type hints required for public functions

**Python Version:**
- Minimum: Python 3.10+
- Use modern type hints: `from typing import *`
- Use `str | None` instead of `Optional[str]` (union syntax)

## Import Organization

**Order:**
1. Standard library imports (`import os`, `import sys`)
2. Third-party imports (`import discord`, `import numpy`)
3. Local imports (`from src.memory import Storage`)
4. Blank line between each group

**Example:**
```python
import asyncio
import json
from pathlib import Path
from typing import Optional

import discord
from dotenv import load_dotenv

from src.memory import ConversationStorage
from src.models import ModelManager
```

**Path Aliases:**
- Use relative imports from `src/` root
- Avoid deep relative imports (no `../../../`)
- Example: `from src.safety import SandboxExecutor` not `from ...safety import SandboxExecutor`

## Error Handling

**Patterns:**
- Define domain-specific exceptions in `src/exceptions.py`
- Use exception hierarchy (base `MaiException`, specific subclasses)
- Always include context in exceptions (error code, details, suggestions)
- Example:

```python
class MaiException(Exception):
    """Base exception for Mai framework."""
    def __init__(self, code: str, message: str, details: dict | None = None):
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(f"[{code}] {message}")

class ModelError(MaiException):
    """Raised when model inference fails."""
    pass

class MemoryError(MaiException):
    """Raised when memory operations fail."""
    pass
```

- Log before raising (see Logging section)
- Use context managers for cleanup (async context managers for async code)
- Never catch bare `Exception` - catch specific exceptions

## Logging

**Framework:** `logging` module (Python standard library)

**Patterns:**
- Create logger per module: `logger = logging.getLogger(__name__)`
- Log levels guide:
  - `DEBUG`: Detailed diagnostic info (token counts, decision trees)
  - `INFO`: Significant operational events (conversation started, model loaded)
  - `WARNING`: Unexpected but handled conditions (fallback triggered, retry)
  - `ERROR`: Failed operation (model error, memory access failed)
  - `CRITICAL`: System-level failures (cannot recover)
- Structured logging preferred (include operation context)
- Example:

```python
import logging

logger = logging.getLogger(__name__)

async def invoke_model(prompt: str, model: str) -> str:
    logger.debug(f"Invoking model={model} with token_count={len(prompt.split())}")
    try:
        response = await model_manager.generate(prompt)
        logger.info(f"Model response generated, length={len(response)}")
        return response
    except ModelError as e:
        logger.error(f"Model invocation failed: {e.code}", exc_info=True)
        raise
```

## Comments

**When to Comment:**
- Complex logic requiring explanation (multi-step algorithms, non-obvious decisions)
- Important context that code alone cannot convey (why a workaround exists)
- Do NOT comment obvious code (`x = 1  # set x to 1` is noise)
- Do NOT duplicate what the code already says

**JSDoc/Docstrings:**
- Use Google-style docstrings for all public functions/classes
- Include return type even if type hints exist (for readability)
- Example:

```python
async def get_memory_context(
    query: str,
    max_tokens: int = 2000,
) -> str:
    """Retrieve relevant memory context for a query.

    Performs vector similarity search on conversation history,
    compresses results to fit token budget, and returns formatted context.

    Args:
        query: The search query for memory retrieval.
        max_tokens: Maximum tokens in returned context (default 2000).

    Returns:
        Formatted memory context as markdown-structured string.

    Raises:
        MemoryError: If database query fails or storage is corrupted.
    """
```

## Function Design

**Size:**
- Target: Functions under 50 lines (hard limit: 100 lines)
- Break complex logic into smaller helper functions
- One responsibility per function (single responsibility principle)

**Parameters:**
- Maximum 4 positional parameters
- Use keyword-only arguments for optional params: `def func(required, *, optional=None)`
- Use dataclasses or TypedDict for complex parameter groups
- Example:

```python
# Good: Clear structure
async def approve_change(
    change_id: str,
    *,
    reviewer_id: str,
    decision: Literal["approve", "reject"],
    reason: str | None = None,
) -> None:
    pass

# Bad: Too many params
async def approve_change(change_id, reviewer_id, decision, reason, timestamp, context, metadata):
    pass
```

**Return Values:**
- Explicitly return values (no implicit `None` returns unless documented)
- Use `Optional[T]` or `T | None` in type hints for nullable returns
- Prefer returning data objects over tuples: return `Result` not `(status, data, error)`
- Async functions return awaitable, not callbacks

## Module Design

**Exports:**
- Define `__all__` in each module to be explicit about public API
- Example in `src/memory/__init__.py`:

```python
from src.memory.storage import ConversationStorage
from src.memory.compression import MemoryCompressor

__all__ = ["ConversationStorage", "MemoryCompressor"]
```

**Barrel Files:**
- Use `__init__.py` to export key classes/functions from submodules
- Keep import chains shallow (max 2 levels deep)
- Example structure:
  ```
  src/
  ├── memory/
  │   ├── __init__.py (exports Storage, Compressor)
  │   ├── storage.py
  │   └── compression.py
  ```

**Async/Await:**
- All I/O operations (database, API calls, file I/O) must be async
- Use `asyncio` for concurrency, not threading
- Async context managers for resource management:

```python
async def process_request(prompt: str) -> str:
    async with model_manager.get_session() as session:
        response = await session.generate(prompt)
        return response
```

## Type Hints

**Requirements:**
- All public function signatures must have type hints
- Use `from __future__ import annotations` for forward references
- Prefer union syntax: `str | None` over `Optional[str]`
- Use `Literal` for string enums: `Literal["approve", "reject"]`
- Example:

```python
from __future__ import annotations
from typing import Literal

def evaluate_risk(code: str) -> Literal["LOW", "MEDIUM", "HIGH", "BLOCKED"]:
    """Evaluate code risk level."""
    pass
```

## Configuration

**Pattern:**
- Use YAML for human-editable config files
- Use environment variables for secrets (never commit `.env`)
- Validation at import time (fail fast if config invalid)
- Example:

```python
# config.py
import os
from pathlib import Path

class Config:
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    MODELS_PATH = Path(os.getenv("MODELS_PATH", "~/.mai/models")).expanduser()
    MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "8000"))

    # Validate on import
    if not MODELS_PATH.exists():
        raise RuntimeError(f"Models path does not exist: {MODELS_PATH}")
```

---

*Convention guide: 2026-01-26*
*Status: Prescriptive for Mai v1 implementation*

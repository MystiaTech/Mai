# Testing Patterns

**Analysis Date:** 2026-01-26

## Status

**Note:** This codebase is in planning phase. No tests have been written yet. These patterns are **prescriptive** for the Mai project and should be applied from the first test file forward.

## Test Framework

**Runner:**
- **pytest** - Test discovery and execution
- Version: Latest stable (6.x or higher)
- Config: `pytest.ini` or `pyproject.toml` (create with initial setup)

**Assertion Library:**
- Built-in `assert` statements
- `pytest` fixtures for setup/teardown
- `pytest.raises()` for exception testing

**Run Commands:**
```bash
pytest                          # Run all tests in tests/ directory
pytest -v                       # Verbose output with test names
pytest -k "test_memory"         # Run tests matching pattern
pytest --cov=src                # Generate coverage report
pytest --cov=src --cov-report=html  # Generate HTML coverage
pytest -x                       # Stop on first failure
pytest -s                       # Show print output during tests
```

## Test File Organization

**Location:**
- **Co-located pattern**: Test files live next to source files
- Structure: `src/[module]/test_[component].py`
- All tests in a single directory: `tests/` with mirrored structure

**Recommended pattern for Mai:**
```
src/
├── memory/
│   ├── __init__.py
│   ├── storage.py
│   └── test_storage.py          # Co-located tests
├── models/
│   ├── __init__.py
│   ├── manager.py
│   └── test_manager.py
└── safety/
    ├── __init__.py
    ├── sandbox.py
    └── test_sandbox.py
```

**Naming:**
- Test files: `test_*.py` or `*_test.py`
- Test classes: `TestComponentName`
- Test functions: `test_specific_behavior_with_context`
- Example: `test_retrieves_conversation_history_within_token_limit`

**Test Organization:**
- One test class per component being tested
- Group related tests in a single class
- One assertion per test (or tightly related assertions)

## Test Structure

**Suite Organization:**
```python
import pytest
from src.memory.storage import ConversationStorage

class TestConversationStorage:
    """Test suite for ConversationStorage."""

    @pytest.fixture
    def storage(self) -> ConversationStorage:
        """Provide a storage instance for testing."""
        return ConversationStorage(path=":memory:")  # Use in-memory DB

    @pytest.fixture
    def sample_conversation(self) -> dict:
        """Provide sample conversation data."""
        return {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ]
        }

    def test_stores_and_retrieves_conversation(self, storage, sample_conversation):
        """Test that conversations can be stored and retrieved."""
        conversation_id = storage.store(sample_conversation)
        retrieved = storage.get(conversation_id)
        assert retrieved == sample_conversation

    def test_raises_error_on_missing_conversation(self, storage):
        """Test that missing conversations raise appropriate error."""
        with pytest.raises(MemoryError):
            storage.get("nonexistent_id")
```

**Patterns:**

- **Setup pattern**: Use `@pytest.fixture` for setup, avoid `setUp()` methods
- **Teardown pattern**: Use fixture cleanup (yield pattern)
- **Assertion pattern**: One logical assertion per test (may involve multiple `assert` statements on related data)

```python
@pytest.fixture
def model_manager():
    """Set up model manager and clean up after test."""
    manager = ModelManager()
    manager.initialize()
    yield manager
    manager.shutdown()  # Cleanup

def test_loads_available_models(model_manager):
    """Test model discovery and loading."""
    models = model_manager.list_available()
    assert len(models) > 0
    assert all(isinstance(m, str) for m in models)
```

## Async Testing

**Pattern:**
```python
import pytest
import asyncio

@pytest.mark.asyncio
async def test_async_model_invocation():
    """Test async model inference."""
    manager = ModelManager()
    response = await manager.generate("test prompt")
    assert len(response) > 0
    assert isinstance(response, str)

@pytest.mark.asyncio
async def test_concurrent_memory_access():
    """Test that memory handles concurrent access."""
    storage = ConversationStorage()
    tasks = [
        storage.store({"id": i, "text": f"msg {i}"})
        for i in range(10)
    ]
    ids = await asyncio.gather(*tasks)
    assert len(ids) == 10
```

- Use `@pytest.mark.asyncio` decorator
- Use `async def` for test function signature
- Use `await` for async calls
- Can mix async fixtures and sync fixtures

## Mocking

**Framework:** `unittest.mock` (Python standard library)

**Patterns:**

```python
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import pytest

def test_handles_model_error():
    """Test error handling when model fails."""
    mock_model = Mock()
    mock_model.generate.side_effect = RuntimeError("Model offline")

    manager = ModelManager(model=mock_model)
    with pytest.raises(ModelError):
        manager.invoke("prompt")

@pytest.mark.asyncio
async def test_retries_on_transient_failure():
    """Test retry logic for transient failures."""
    mock_api = AsyncMock()
    mock_api.call.side_effect = [
        Exception("Temporary failure"),
        "success"
    ]

    result = await retry_with_backoff(mock_api.call, max_retries=2)
    assert result == "success"
    assert mock_api.call.call_count == 2

@patch("src.models.manager.requests.get")
def test_fetches_model_list(mock_get):
    """Test fetching model list from API."""
    mock_get.return_value.json.return_value = {"models": ["model1", "model2"]}

    manager = ModelManager()
    models = manager.get_remote_models()
    assert models == ["model1", "model2"]
```

**What to Mock:**
- External API calls (Discord, LMStudio API)
- Database operations (SQLite in production, use in-memory for tests)
- File I/O (use temporary directories)
- Slow operations (model inference can be stubbed)
- System resources (CPU, RAM monitoring)

**What NOT to Mock:**
- Core business logic (the logic you're testing)
- Data structure operations (dict, list operations)
- Internal module calls within the same component
- Internal helper functions

## Fixtures and Factories

**Test Data Pattern:**

```python
# conftest.py - shared fixtures
import pytest
from pathlib import Path
from src.memory.storage import ConversationStorage

@pytest.fixture
def temp_db():
    """Provide a temporary SQLite database."""
    db_path = Path("/tmp/test_mai.db")
    yield db_path
    if db_path.exists():
        db_path.unlink()

@pytest.fixture
def conversation_factory():
    """Factory for creating test conversations."""
    def _make_conversation(num_messages: int = 3) -> dict:
        messages = []
        for i in range(num_messages):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({
                "role": role,
                "content": f"Message {i+1}",
                "timestamp": f"2026-01-26T{i:02d}:00:00Z"
            })
        return {"messages": messages}
    return _make_conversation

def test_stores_long_conversation(temp_db, conversation_factory):
    """Test storing conversations with many messages."""
    storage = ConversationStorage(path=temp_db)
    long_convo = conversation_factory(num_messages=100)

    conv_id = storage.store(long_convo)
    retrieved = storage.get(conv_id)
    assert len(retrieved["messages"]) == 100
```

**Location:**
- Shared fixtures: `tests/conftest.py` (pytest auto-discovers)
- Component-specific fixtures: In test files or subdirectory `conftest.py` files
- Factories: In `tests/factories.py` or within `conftest.py`

## Coverage

**Requirements:**
- **Target: 80% code coverage minimum** for core modules
- Critical paths (safety, memory, inference): 90%+ coverage
- UI/CLI: 70% (lower due to interaction complexity)

**View Coverage:**
```bash
pytest --cov=src --cov-report=term-missing
pytest --cov=src --cov-report=html
# Then open htmlcov/index.html in browser
```

**Configure in `pyproject.toml`:**
```toml
[tool.pytest.ini_options]
testpaths = ["src", "tests"]
addopts = "--cov=src --cov-report=term-missing --cov-report=html"
```

## Test Types

**Unit Tests:**
- Scope: Single function or class method
- Dependencies: Mocked
- Speed: Fast (<100ms per test)
- Location: `test_component.py` in source directory
- Example: `test_tokenizer_splits_input_correctly`

**Integration Tests:**
- Scope: Multiple components working together
- Dependencies: Real services (in-memory DB, local files)
- Speed: Medium (100ms - 1s per test)
- Location: `tests/integration/test_*.py`
- Example: `test_conversation_engine_with_memory_retrieval`

```python
# tests/integration/test_conversation_flow.py
@pytest.mark.asyncio
async def test_full_conversation_with_memory():
    """Test complete conversation flow including memory retrieval."""
    memory = ConversationStorage(path=":memory:")
    engine = ConversationEngine(memory=memory)

    # Store context
    memory.store({"id": "ctx1", "content": "User prefers Python"})

    # Have conversation
    response = await engine.chat("What language should I use?")

    # Verify context was used
    assert "Python" in response or "python" in response.lower()
```

**E2E Tests:**
- Scope: Full system end-to-end
- Framework: **Not required for v1** (added in v2)
- Would test: CLI input → Model → Discord output
- Deferred until Discord/CLI interfaces complete

## Common Patterns

**Error Testing:**
```python
def test_invalid_input_raises_validation_error():
    """Test that validation catches malformed input."""
    with pytest.raises(ValueError) as exc_info:
        storage.store({"invalid": "structure"})
    assert "missing required field" in str(exc_info.value)

def test_logs_error_details():
    """Test that errors log useful debugging info."""
    with patch("src.logger") as mock_logger:
        try:
            risky_operation()
        except OperationError:
            pass
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args
        assert "operation_id" in str(call_args)
```

**Performance Testing:**
```python
def test_memory_retrieval_within_performance_budget(benchmark):
    """Test that memory queries complete within time budget."""
    storage = ConversationStorage()
    query = "what did we discuss earlier"

    result = benchmark(storage.retrieve_similar, query)
    assert len(result) > 0

# Run with: pytest --benchmark-only
```

**Data Validation Testing:**
```python
@pytest.mark.parametrize("input_val,expected", [
    ("hello", "hello"),
    ("HELLO", "hello"),
    ("  hello  ", "hello"),
    ("", ValueError),
])
def test_normalizes_input(input_val, expected):
    """Test input normalization with multiple cases."""
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            normalize(input_val)
    else:
        assert normalize(input_val) == expected
```

## Configuration

**pytest.ini (create at project root):**
```ini
[pytest]
testpaths = src tests
addopts = -v --tb=short --strict-markers
markers =
    asyncio: marks async tests
    slow: marks slow tests
    integration: marks integration tests
```

**Alternative: pyproject.toml:**
```toml
[tool.pytest.ini_options]
testpaths = ["src", "tests"]
addopts = "-v --tb=short"
markers = [
    "asyncio: async test",
    "slow: slow test",
    "integration: integration test",
]
```

## Test Execution in CI/CD

**GitHub Actions workflow (when created):**
```yaml
- name: Run tests
  run: pytest --cov=src --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    files: ./coverage.xml
```

---

*Testing guide: 2026-01-26*
*Status: Prescriptive for Mai v1 implementation*

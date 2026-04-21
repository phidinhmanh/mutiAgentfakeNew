---
name: testing
description: Testing conventions and best practices
type: rules
---

# Testing Conventions

## Testing Philosophy

> "Write tests not to prove that code works, but to prove that code doesn't fail." — Aim to break your code through thorough testing.

## Test File Structure

```
tests/
├── conftest.py           # Shared fixtures
├── unit/
│   ├── test_config.py
│   └── test_models.py
├── integration/
│   └── test_api.py
└── fixtures/
    └── sample_data.py
```

## Test Naming

```python
def test_function_name_when_condition_then_expected():
    """Descriptive test names following Gherkin-style."""

def test_user_repository_when_duplicate_email_then_raises_error():
    pass

def test_json_exporter_with_valid_data_returns_json_string():
    pass
```

## Arrange-Act-Assert (AAA) Pattern

```python
def test_calculate_total_with_multiple_items():
    # Arrange
    items = [
        Price(amount=10.0, tax_rate=0.1),
        Price(amount=20.0, tax_rate=0.1),
    ]
    config = Config(tax_rate=0.1)

    # Act
    result = calculate_total(items, config)

    # Assert
    assert result == 33.0  # (10 + 20) * 1.1
    assert isinstance(result, float)
```

## Pytest Fixtures

```python
import pytest
from myapp.models import User, Config

@pytest.fixture
def sample_user():
    """Create a sample user for tests."""
    return User(name="Test User", email="test@example.com")

@pytest.fixture
def sample_config():
    """Create a test configuration."""
    return Config(debug=True, max_retries=3)

@pytest.fixture
def mock_api():
    """Mock external API calls."""
    with patch("myapp.services.api") as mock:
        mock.return_value = {"status": "ok"}
        yield mock
```

## Test Doubles (Mocks/Stubs)

```python
from unittest.mock import Mock, patch, MagicMock

def test_email_sender_sends_correct_data():
    # Stub: Pre-programmed responses
    mailer = Mock()
    mailer.send.return_value = True

    # Use stub
    sender = EmailSender(mailer)
    result = sender.send("test@example.com", "Hello")

    # Verify
    assert result is True
    mailer.send.assert_called_once_with(
        to="test@example.com",
        subject="Hello"
    )

def test_cache_gets_from_backend():
    # Mock: Full mock object
    cache = MagicMock()
    cache.get.side_effect = [None, {"data": "cached"}]

    service = DataService(cache=cache)

    # First call - miss, fetches from backend
    result1 = service.get_data("key1")
    assert result1 is None

    # Second call - hit from cache
    result2 = service.get_data("key1")
    assert result2 == {"data": "cached"}
```

## Property-Based Testing (Hypothesis)

```python
from hypothesis import given, strategies as st

@given(st.lists(st.integers(min_value=1, max_value=1000), min_size=1))
def test_total_always_positive(numbers):
    """Total of positive integers is always positive."""
    assert sum(numbers) > 0

@given(
    st.text(min_size=1, max_size=100),
    st.integers(min_value=0, max_value=100)
)
def test_string_repeat_bounds(text, count):
    """String repetition respects max bounds."""
    result = safe_repeat(text, count)
    # Should not exceed reasonable bounds
    assert len(result) <= len(text) * 100 + count * 2
```

## Integration Tests

```python
import pytest
from myapp.database import Database

@pytest.fixture
def test_db():
    """Create isolated test database."""
    db = Database(":memory:")
    db.connect()
    db.init_schema()

    yield db

    db.close()

def test_create_and_retrieve_user(test_db):
    """Integration test for database operations."""
    # Create
    user_id = test_db.users.create(
        name="Test User",
        email="test@example.com"
    )

    # Retrieve
    user = test_db.users.get_by_id(user_id)

    assert user is not None
    assert user.name == "Test User"
    assert user.email == "test@example.com"
```

## Coverage Requirements

| Coverage Type | Minimum | Target |
|--------------|---------|--------|
| Line coverage | 80% | 90% |
| Branch coverage | 70% | 80% |
| Function coverage | 100% | 100% |

```bash
# Check coverage
pytest --cov=src --cov-report=term-missing tests/

# Generate HTML report
pytest --cov=src --cov-report=html tests/
```

## Pytest Configuration (pyproject.toml)

```toml
[tool.pytest.ini_options]
minversion = "8.0"
testpaths = ["tests"]
pythonpath = ["."]

# Markers
markers = [
    "slow: marks tests as slow",
    "integration: marks integration tests",
    "unit: marks unit tests",
]

# Coverage
addopts = [
    "--strict-markers",
    "--tb=short",
    "-v",
]
```

## Test Execution

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src tests/

# Run specific marker
poetry run pytest -m unit

# Run specific test file
poetry run pytest tests/unit/test_config.py

# Run tests matching pattern
poetry run pytest -k "test_user"
```

## CI/CD Integration

```yaml
# GitHub Actions
- name: Run tests
  run: |
    poetry install
    poetry run pytest --cov=src --cov-fail-under=80
```

## Anti-Patterns to Avoid

```python
# ❌ Don't test implementation details
def test_internal_method():
    obj = MyClass()
    assert obj._internal_state == "expected"  # Fragile!

# ✅ Test behavior, not implementation
def test_returns_sorted_results():
    obj = MyClass()
    result = obj.get_sorted_items()
    assert result == [1, 2, 3, 4, 5]

# ❌ Don't use sleep for timing
def test_async_behavior():
    time.sleep(5)  # Flaky!
    assert result == "done"

# ✅ Use proper async testing
@pytest.mark.asyncio
async def test_async_behavior():
    result = await process()
    assert result == "done"
```

## Edge Case Testing Checklist

- [ ] Empty inputs
- [ ] Single item
- [ ] Maximum/minimum values
- [ ] Null/None values
- [ ] Negative numbers
- [ ] Special characters
- [ ] Unicode characters
- [ ] Very long strings
- [ ] Concurrent access
- [ ] Timeout scenarios
- [ ] Network failures
- [ ] Disk full scenarios

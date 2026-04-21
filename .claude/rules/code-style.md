---
name: code-style
description: Code style and formatting guidelines cho Python project
type: rules
---

# Code Style Guidelines

## Python Style Guide (PEP 8 + Modern Standards)

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Variables | snake_case | `user_name`, `total_count` |
| Functions | snake_case | `def get_user_data()` |
| Classes | PascalCase | `class UserRepository` |
| Constants | UPPER_SNAKE | `MAX_RETRIES`, `DEFAULT_TIMEOUT` |
| Private | _prefix | `_internal_state` |
| Type aliases | PascalCase suffix `_T` | `UserIdT = int` |

### Type Hints (Mandatory)

```python
# Always use type hints
def process_data(items: list[dict[str, Any]], config: Config) -> Result:
    pass

# Generic types
from typing import Sequence, Optional

def find_by_id(items: Sequence[Item], id: int) -> Optional[Item]:
    pass

# Modern Python 3.10+ union syntax
def get_name() -> str | None:
    pass
```

### Docstrings

```python
def calculate_total(items: list[Price]) -> float:
    """Calculate total price including tax.

    Args:
        items: List of price objects to sum
        config: Configuration with tax rate

    Returns:
        Total amount with tax applied

    Raises:
        ValueError: If items is empty
    """
    if not items:
        raise ValueError("Items cannot be empty")
    return sum(item.amount for item in items)
```

## Formatting Rules

### Line Length

- Maximum **88 characters** (Ruff default, aligned with Black)
- Use line continuation for long lines:

```python
# ✅ Good
result = some_function(
    arg1="long value",
    arg2="another long value",
    arg3="yet another",
)

# ❌ Bad
result = some_function(arg1="long value", arg2="another long value")
```

### Imports

```python
# 1. Standard library
import os
import sys
from typing import Optional

# 2. Third party
import click
from pydantic import BaseModel

# 3. Local application
from myapp.core.config import Settings
from myapp.utils import helpers

# 4. Separate groups with blank line
# 5. Sort alphabetically within group
```

### Whitespace

```python
# ✅ Good
x = 1
y = 2

def foo():
    pass

class Bar:
    def method(self):
        pass

# ❌ Bad
x=1
y=2

def foo ():
    pass

class Bar:
    def method (self):
        pass
```

## Clean Code Principles

### SOLID Principles

1. **S**ingle Responsibility: Each class/function has one job
2. **O**pen/Closed: Open for extension, closed for modification
3. **L**iskov Substitution: Subtypes substitutable for base types
4. **I**nterface Segregation: Many small interfaces
5. **D**ependency Inversion: Depend on abstractions

### DRY (Don't Repeat Yourself)

```python
# ❌ Repeated
if user.role == "admin":
    grant_access()
elif user.role == "manager":
    grant_access()
elif user.role == "editor":
    grant_access()

# ✅ Reusable
ADMIN_ROLES = {"admin", "manager", "editor"}
if user.role in ADMIN_ROLES:
    grant_access()
```

### Error Handling

```python
# ✅ Specific exceptions
def read_config(path: Path) -> Config:
    try:
        return Config.parse_file(path)
    except FileNotFoundError:
        raise ConfigError(f"Config not found: {path}")
    except ValidationError as e:
        raise ConfigError(f"Invalid config: {e}")

# ❌ Bare except (never do this)
try:
    do_something()
except:
    pass
```

## Project-Specific Rules

### File Organization

- Maximum **500 lines** per file
- Split large modules into submodules
- Group related functions into classes or modules

### Comments

```python
# Explain WHY, not WHAT
# ❌ Bad: "Increment counter"
# ✅ Good: "Retry counter for rate limit handling"

# Use TODO for future work
# TODO(username): Implement caching layer
```

## Linting Commands

```bash
# Check code style
ruff check src/ tests/

# Auto-fix issues
ruff check --fix src/

# Format code
ruff format src/
```

## Pre-commit Hook

Add to `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: ruff-format
        name: ruff format
        entry: ruff format
        language: system
        types: [python]
      - id: ruff-check
        name: ruff check
        entry: ruff check
        language: system
        types: [python]
```
---
name: security
description: Security guidelines cho Python project
type: rules
---

# Security Guidelines

## OWASP Top 10 Principles

### 1. Injection Prevention

```python
# ❌ KHÔNG BAO GIỜ: SQL string concatenation
query = f"SELECT * FROM users WHERE name = '{user_input}'"

# ✅ ĐÚNG: Parameterized queries
cursor.execute("SELECT * FROM users WHERE name = ?", (user_input,))
```

### 2. Sensitive Data Handling

```python
# Environment variables cho secrets
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    api_key: str  # Từ env var, không hardcode
    db_password: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
```

### 3. Input Validation

```python
from pydantic import BaseModel, validator

class UserInput(BaseModel):
    name: str
    age: int

    @validator("age")
    def validate_age(cls, v):
        if v < 0 or v > 150:
            raise ValueError("Age must be realistic")
        return v

    @validator("name")
    def validate_name(cls, v):
        if not v or len(v) > 100:
            raise ValueError("Invalid name")
        return v.strip()
```

### 4. Secure Defaults

```python
# Default = secure
class Config:
    debug: bool = False  # Không leak stack traces
    allow_origins: list[str] = []  # Empty = deny all
    require_auth: bool = True
```

### 5. Secrets Management

```python
# ❌ KHÔNG BAO GIỜ:
API_KEY = "sk-1234567890abcdef"

# ✅ SỬ DỤNG:
# - Environment variables
# - .env file (không commit!)
# - Secret manager (AWS Secrets Manager, HashiCorp Vault)
```

## Security Checklist

- [ ] Không hardcode credentials
- [ ] Validate all external input
- [ ] Use parameterized queries
- [ ] Enable HTTPS only
- [ ] Set secure cookie flags
- [ ] Sanitize file paths (path traversal prevention)
- [ ] Rate limiting cho APIs
- [ ] Log không chứa sensitive data

## Path Traversal Prevention

```python
from pathlib import Path
import os

def safe_read_file(base_dir: Path, user_path: str) -> str:
    # Normalize và resolve
    requested = (base_dir / user_path).resolve()
    # Ensure nằm trong base_dir
    if not requested.is_relative_to(base_dir):
        raise ValueError("Access denied")
    return requested.read_text()
```

## Dependency Security

```bash
# Audit dependencies
pip audit
# hoặc
poetry check
```

## Logging Security

```python
import logging
import re

# Sanitize sensitive data trước khi log
def sanitize_for_log(data: dict) -> dict:
    sensitive_keys = {"password", "api_key", "token", "secret"}
    return {
        k: "***REDACTED***" if k.lower() in sensitive_keys else v
        for k, v in data.items()
    }
```

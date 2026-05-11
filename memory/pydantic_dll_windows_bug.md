---
name: pydantic_dll_windows_bug
description: Pydantic v2 pydantic-core .pyd DLL blocked by Windows Application Control policy — resolved by replacing pydantic with dataclasses
type: reference
---
# Pydantic DLL Block on Windows — ROOT CAUSE & RESOLUTION

## Bug Description
```
ImportError: DLL load failed while importing _pydantic_core: An Application Control policy has blocked this file
```

**Symptom**: Python crashes immediately on startup when importing any module that uses pydantic, including `trust_agents.config`, `fake_news_detector.config`, etc.

**Trigger**: `import pydantic` or `from pydantic import BaseModel` — even a simple `import pydantic; print(pydantic.__version__)` fails.

## Root Cause

**Windows Application Control Policy** (Windows Defender Application Control / Code Integrity) blocks execution of unsigned binary DLLs in specific paths. The `pydantic-core` package includes a compiled extension `_pydantic_core.cp313-win_amd64.pyd` (a Windows DLL). On Manh's machine, this DLL is blocked.

**Why downgrading doesn't work**: ALL versions of `pydantic-core` (2.x series) use the same binary extension mechanism. Even `pydantic-core==2.26.0` still has a `.pyd` DLL that gets blocked. Pydantic v1 also loads `pydantic-core` v1 which also has a binary. There is NO version of pydantic that works without a binary extension on CPython 3.13 + Windows.

**Why the policy blocks it**: The `.pyd` file is a compiled Windows DLL loaded by Python's `import` machinery. When Windows loads the DLL, the Application Control policy evaluates it and blocks it because the DLL is not signed / not in an approved list.

## Solution: Zero-Pydantic Strategy

Replace all pydantic usage with Python Standard Library alternatives:

| File | Before | After | Notes |
|------|--------|-------|-------|
| `src/trust_agents/config.py` | `BaseModel` + `Field` | `@dataclass` | No validation needed, just config data |
| `src/fake_news_detector/config.py` | `BaseSettings` | `@dataclass` + `os.environ` | Replace `SettingsConfigDict` env loading with manual `os.getenv` |
| `src/trust_agents/llm/gemini_langchain.py` | `ConfigDict`, `Field`, `PrivateAttr` | `@dataclass` + instance attributes | Remove `arbitrary_types_allowed=True`, replace `PrivateAttr` with regular attribute |
| `src/trust_agents/llm/gemini_client.py` | `BaseModel` | `@dataclass` | Dead code (never imported) — DELETE |
| `src/trust_agents/llm/groq_client.py` | `BaseModel` | `@dataclass` | Dead code (never imported) — DELETE |
| `pyproject.toml` | `pydantic-settings>=2.0.0` | REMOVED | `python-dotenv` already present for env loading |

**Why dataclass works here**: All pydantic models in this project are used ONLY for configuration/data storage. None use pydantic's runtime validation features. `dataclass` provides identical functionality (typed fields, default values, `__init__`, `__repr__`) without any compiled dependencies.

## What Was Verified

- `gemini_client.py` and `groq_client.py` are dead code — not imported anywhere in the codebase. Safe to delete.
- `gemini_langchain.py` uses `PrivateAttr` for the `_client` cache — replaced with a regular instance attribute.
- `fake_news_detector/config.py` Settings are loaded at module level with `settings = Settings()` — replaced with `dataclass` + `load_dotenv()`.
- The `LLMConfig.from_env()` classmethod reads from `os.getenv` directly — trivial to port.

## Prevention

- Never add `pydantic` back as a dependency. Use `dataclasses` for config/data classes.
- If runtime validation is needed, use `beartype` (pure Python) or manual validation functions instead.
- Before adding a new dependency that has compiled extensions, test on a Windows environment with strict Application Control policies.
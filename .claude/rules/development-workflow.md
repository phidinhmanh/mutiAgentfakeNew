# Development Workflow Rule

This rule defines the mandatory verification steps and coding patterns to ensure stability on the Multi-Agent Fake News pipeline, specifically for Windows development environments.

## 1. Stabilization & Verification

Every non-trivial change MUST follow this sequence:

1.  **Smoke Check**: Run `uv run scripts/smoke_test.py` to verify that environment/dependency state is still valid.
2.  **Unit Tests**: Run tests for the specific module being modified.
    - `uv run pytest tests/unit/test_FILENAME.py`
3.  **Lint Check**: Run Ruff to ensure formatting and style compliance.
    - `uv run ruff check .`
    - `uv run ruff format .`

## 2. Windows Compatibility

*   **Lazy GenAI Imports**: Do NOT import `google.genai` or `langgraph` at the top level of shared modules (like factories or configs). Use lazy loading functions to avoid `pydantic-core` DLL load failures that can happen during early initialization on Windows.
*   **Path Handling**: Always use `pathlib.Path` or `os.path.join` for file operations. Avoid hardcoded `/` or `\`.

## 3. Testing Patterns

*   **Mocking**: Never call real LLM APIs or Web Search APIs in unit tests.
*   **Patch Paths**: When patching a function, patch it in the module where it is **imported/used**, not where it is defined, if that module uses `from module import function`.
*   **Cache Invalidation**: If a module uses global state or caches (like the retrieval service), ensure tests clear this state in a fixture or teardown.

## 4. Labeling & Consistency

*   **Verdict Labels**: Always use `REAL`, `FAKE`, `UNCERTAIN`, or `UNKNOWN`. Avoid legacy labels like `true`, `false`, `unverifiable`, or `unsupported`.
*   **Confidence**: Always normalize to a float between `0.0` and `1.0`.

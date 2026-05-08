# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Multi-Agent Fake News Detection - Architecture & Guidelines

## Project Overview

Vietnamese fake news detection with two parallel analysis paths:
- `fake_news_detector`: baseline PhoBERT classifier, stylistic features, and hybrid RAG utilities.
- `trust_agents`: multi-agent fact-checking pipeline that extracts claims, retrieves evidence, verifies each claim, and generates explanations.
- `api`: FastAPI backend providing REST endpoints for analysis and fact-checking.

## Common Commands

```bash
# Install dependencies
uv sync

# Install with dev tools
uv sync --all-extras

# Run the FastAPI backend
uv run python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Run the interactive terminal runner
uv run python scripts/interactive_runner.py

# Download ViFactCheck data and build FAISS index
uv run python scripts/download_data.py --build-index --max-samples 5000

# Run the TRUST orchestrator directly
uv run python -m trust_agents.orchestrator --text "Nội dung cần kiểm tra" --top-k 5

# Run the research orchestrator directly
uv run python -m trust_agents.orchestrator_research --text "Nội dung cần kiểm tra"

# Run all tests
uv run pytest tests/ -v

# Run a single test file
uv run pytest tests/unit/test_citation_checker.py -v

# Run a single test
uv run pytest tests/integration/test_orchestrator.py -k process_text_with_multiple_claims -v

# Run only integration tests
uv run pytest tests/integration/ -v

# Run the system smoke test (Quick verification of environment/imports)
uv run scripts/smoke_test.py

# Lint and format
uv run ruff check .
uv run ruff format .

# Type check
uv run mypy src/
```

## Architecture

### 1. App entrypoints

- `src/api/main.py`: FastAPI entrypoint. Main REST API for analysis and SSE streaming logs.
- `scripts/interactive_runner.py`: terminal workflow for running baseline-only, TRUST-only, or full analysis without the UI.
- `src/trust_agents/orchestrator.py`: main 4-agent fact-checking orchestrator.
- `src/trust_agents/orchestrator_research.py`: alternate research-grade orchestrator with expanded pipeline logic.

### 2. App service layer (`src/fake_news_detector`)

The `fake_news_detector` package handles core analysis and retrieval:

- `application/analysis_service.py`: shared article-analysis flow; handles long-text summarization, baseline scoring, stylistic features, and TRUST result shaping.
- `application/index_service.py`: sample-data loading and FAISS index build flows.
- `config.py`: central Pydantic settings loaded from `.env`; controls model names, FAISS path, retrieval thresholds, and search provider.
- `models/baseline.py`: PhoBERT-based fake-news classifier with sliding-window inference.
- `models/stylistic.py`: handcrafted stylistic features shown alongside model output.
- `rag/vector_store.py`: FAISS index lifecycle and similarity search.
- `rag/retriever.py`: compatibility wrapper over the shared retrieval core; queries FAISS first, then falls back to web search.
- `rag/web_search.py`: provider-specific web retrieval (`serper` or `tavily`).
- `utils/citation_checker.py`: validates whether quoted reasoning is grounded in retrieved evidence.

### 3. Shared retrieval/parsing seams

- `src/shared_fact_checking/retrieval/policy.py`: shared confidence scoring and merge/deduplication policy for local + web retrieval results.
- `src/shared_fact_checking/retrieval/service.py`: shared fallback orchestration that decides when web search should augment FAISS results.
- `src/trust_agents/parsing.py`: shared helpers for extracting final message text and parsing JSON-like agent outputs.
- `src/trust_agents/llm/factory.py`: central chat-model factory used by TRUST agents.

### 4. TRUST multi-agent stack (`src/trust_agents`)

The production orchestrator is a 4-step per-claim pipeline:
1. `agents/claim_extractor.py`: extracts factual claims from the input text
2. `agents/evidence_retrieval.py`: retrieves supporting/contradicting evidence for each claim
3. `agents/verifier.py`: assigns verdict + confidence for each claim
4. `agents/explainer.py`: produces a readable summary/explanation for the verified claim

### 5. LLM/provider configuration

- `src/fake_news_detector/config.py`: app/runtime settings for retrieval, baseline, and API behavior.
- `src/trust_agents/config.py`: provider/model selection for TRUST agent LLM backends.

### 6. Testing layout

- `tests/unit/`: focused tests for baseline models, retrieval, parsing, LLM factory, citation checking, and service-level behavior
- `tests/integration/`: orchestrator-level tests for TRUST pipelines
- `tests/acceptance/`: end-to-end fact-check flow coverage
- `tests/conftest.py`: shared fixtures and mocks used across suites

## Recent Refactor Notes

- Streamlit UI code removed. Project now focuses on REST API (`src/api/`).
- Retrieval fallback/merge logic centralized in `shared_fact_checking`.
- TRUST agent model creation and output parsing centralized.

## Existing Project Rules

Read the rules in `.claude/rules/` before making non-trivial changes:
- `code-style.md`: Python style, naming, typing, Ruff formatting
- `testing.md`: pytest conventions, markers, AAA structure
- `security.md`: input validation, secrets handling, OWASP-oriented guidance
- `git-version-control.md`: branch/commit/PR conventions used by this project

## Stability & Reliability Guidelines

1.  **Mandatory Smoke Test**: Run `uv run scripts/smoke_test.py` after adding any new dependency or changing core imports.
2.  **Test Before Commit**: Run at least the relevant unit tests (`uv run pytest tests/unit/test_FILENAME.py`) before considering a task "done".
3.  **Mock Early**: All LLM and API calls MUST be mocked in unit tests.
4.  **Lazy Imports**: Keep `google-genai` and `langgraph` imports lazy to avoid DLL load hangs on Windows.
5.  **Clean State**: Tests must clear global caches in a teardown or fixture.
6.  **Label Consistency**: Use normalized verdict labels (`REAL`, `FAKE`, `UNCERTAIN`, `UNKNOWN`).

## Environment Notes

```bash
LLM_PROVIDER=google|nvidia|openai|groq
GEMINI_API_KEY=...
SERPER_API_KEY=...
TAVILY_API_KEY=...
```

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Vietnamese fake news detection with two parallel analysis paths:
- `fake_news_detector`: baseline PhoBERT classifier, stylistic features, hybrid RAG utilities, and Streamlit UI
- `trust_agents`: multi-agent fact-checking pipeline that extracts claims, retrieves evidence, verifies each claim, and generates explanations

The current UI runs baseline + stylistic analysis for every article, then chooses either the TRUST pipeline or the legacy single-agent path.

## Common Commands

```bash
# Install dependencies
uv sync

# Install with dev tools
uv sync --all-extras

# Run the Streamlit app
uv run streamlit run src/fake_news_detector/app.py

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

- `src/fake_news_detector/app.py`: Streamlit entrypoint. The UI is now thin: it renders controls/results and delegates analysis/index work to application services.
- `scripts/interactive_runner.py`: terminal workflow for running baseline-only, TRUST-only, or full analysis without the UI.
- `src/trust_agents/orchestrator.py`: main 4-agent fact-checking orchestrator used by the UI.
- `src/trust_agents/orchestrator_research.py`: alternate research-grade orchestrator with expanded pipeline logic.

### 2. App service layer (`src/fake_news_detector`)

The `fake_news_detector` package is split into UI, application services, and model/retrieval infrastructure:

- `application/analysis_service.py`: shared article-analysis flow used by the UI; handles long-text summarization, baseline scoring, stylistic features, TRUST result shaping, and legacy result shaping.
- `application/index_service.py`: sample-data loading and FAISS index build flows used by the sidebar.
- `ui/session.py`: Streamlit session-state defaults and initialization.
- `config.py`: central Pydantic settings loaded from `.env`; controls model names, FAISS path, retrieval thresholds, and search provider.
- `models/baseline.py`: PhoBERT-based fake-news classifier with sliding-window inference.
- `models/stylistic.py`: handcrafted stylistic features shown alongside model output.
- `rag/vector_store.py`: FAISS index lifecycle and similarity search.
- `rag/retriever.py`: compatibility wrapper over the shared retrieval core; queries FAISS first, then falls back to web search when confidence is below `settings.similarity_threshold`.
- `rag/web_search.py`: provider-specific web retrieval (`serper` or `tavily`).
- `utils/citation_checker.py`: validates whether quoted reasoning is grounded in retrieved evidence.
- `agents/`: older single-agent implementation used as a legacy fallback from the Streamlit app.

### 3. Shared retrieval/parsing seams

Recent refactors introduced shared seams to reduce duplicate logic between the Streamlit app path and the TRUST agent path:

- `src/shared_fact_checking/retrieval/policy.py`: shared confidence scoring and merge/deduplication policy for local + web retrieval results.
- `src/shared_fact_checking/retrieval/service.py`: shared fallback orchestration that decides when web search should augment FAISS results.
- `src/trust_agents/parsing.py`: shared helpers for extracting final message text and parsing JSON-like agent outputs.
- `src/trust_agents/llm/factory.py`: central chat-model factory used by TRUST agents instead of duplicating provider/model setup per file.

When changing retrieval semantics, update both wrapper callers and the characterization tests before changing output shapes.

### 4. TRUST multi-agent stack (`src/trust_agents`)

The production orchestrator is a 4-step per-claim pipeline:
1. `agents/claim_extractor.py`: extracts factual claims from the input text
2. `agents/evidence_retrieval.py`: retrieves supporting/contradicting evidence for each claim
3. `agents/verifier.py`: assigns verdict + confidence for each claim
4. `agents/explainer.py`: produces a readable summary/explanation for the verified claim

`TRUSTOrchestrator.process_text()` runs claim extraction once, then loops claim-by-claim through retrieval, verification, and explanation. The orchestrator also normalizes verifier outputs so downstream code sees stable verdict labels (`true`, `false`, `uncertain`) and confidence values in the `0.0..1.0` range.

### 5. LLM/provider configuration

There are two config layers to keep straight:
- `src/fake_news_detector/config.py`: app/runtime settings for retrieval, baseline, and UI behavior
- `src/trust_agents/config.py`: provider/model selection for TRUST agent LLM backends (`google`, `nvidia`, `openai`, `groq`)

If behavior looks inconsistent, check both configs and the corresponding environment variables before changing agent logic.

### 6. Testing layout

- `tests/unit/`: focused tests for baseline models, retrieval, parsing, LLM factory, citation checking, and service-level behavior
- `tests/integration/`: orchestrator-level tests for TRUST pipelines
- `tests/acceptance/`: end-to-end fact-check flow coverage
- `tests/conftest.py`: shared fixtures and mocks used across suites

Useful characterization tests added during the refactor:
- `tests/unit/test_trust_retrieval_tool.py`
- `tests/unit/test_llm_factory.py`
- `tests/unit/test_agent_output_parsing.py`
- `tests/unit/test_app_analysis_service.py`

If you refactor retrieval, agent parsing, or app service boundaries, run these focused tests first before widening to full-suite checks.

## Recent Refactor Notes

The codebase recently went through a phased cleanup to reduce duplication while preserving existing contracts:
- retrieval fallback/merge logic was centralized into `shared_fact_checking`
- TRUST agent model creation and output parsing were centralized
- Streamlit business logic was moved out of `app.py` into `application/` and `ui/`

Because this refactor is still being stabilized, prefer small compatibility-preserving changes over public API rewrites unless the user explicitly asks for broader cleanup.

Current state to be aware of:
- focused unit tests for the refactor seams are passing
- there are still Ruff issues remaining in some touched `src/trust_agents/agents/*.py` and shared files, so a full repo-wide lint pass may fail until those are cleaned up
- when updating tests around `fake_news_detector.app`, patch service-layer symbols in `fake_news_detector.application.analysis_service` when the logic under test lives there

## Existing Project Rules

Read the rules in `.claude/rules/` before making non-trivial changes:
- `code-style.md`: Python style, naming, typing, Ruff formatting
- `testing.md`: pytest conventions, markers, AAA structure
- `security.md`: input validation, secrets handling, OWASP-oriented guidance
- `git-version-control.md`: branch/commit/PR conventions used by this project

The most relevant repo-specific guidance from those rules:
- keep Python files typed and Ruff-formatted
- prefer behavioral tests over implementation-detail tests
- avoid broad destructive git operations on `main`; use feature branches for real git workflows
- treat external input, paths, and secrets as security-sensitive boundaries
- keep modules focused; split oversized files instead of adding more logic to an already large entrypoint

## Environment Notes

Important environment variables used by the current codebase:

```bash
NVIDIA_API_KEY=...
SERPER_API_KEY=...
TAVILY_API_KEY=...
HF_TOKEN=...
LLM_PROVIDER=google|nvidia|openai|groq
GEMINI_API_KEY=...
GOOGLE_API_KEY=...
OPENAI_API_KEY=...
GROQ_API_KEY=...
```

`fake_news_detector.config.Settings` defaults `search_provider` to `serper` and `similarity_threshold` to `0.7`, so weak FAISS matches will trigger web search when keys are present.

## Existing Project Rules

Read the rules in `.claude/rules/` before making non-trivial changes:
- `code-style.md`: Python style, naming, typing, Ruff formatting
- `testing.md`: pytest conventions, markers, AAA structure
- `security.md`: input validation, secrets handling, OWASP-oriented guidance
- `git-version-control.md`: branch/commit/PR conventions used by this project
- `development-workflow.md`: mandatory stabilization and verification steps

## Stability & Reliability Guidelines

To avoid recurring errors and long fix cycles:

1.  **Mandatory Smoke Test**: Run `uv run scripts/smoke_test.py` after adding any new dependency or changing core imports. This catches platform-specific DLL issues on Windows.
2.  **Test Before Commit**: Run at least the relevant unit tests (`uv run pytest tests/unit/test_FILENAME.py`) before considering a task "done".
3.  **Mock Early**: All LLM and API calls MUST be mocked in unit tests. Use the factory patterns in `src/trust_agents/llm/factory.py` to ensure consistent mocking.
4.  **Lazy Imports**: Keep `google-genai` and `langgraph` imports lazy or behind guards if they are not used globally, to avoid DLL load hangs on Windows.
5.  **Clean State**: Tests that use global caches (e.g., `shared_fact_checking.retrieval.service._retrieval_cache`) must clear them in a teardown or fixture to prevent test leakage.
6.  **Label Consistency**: Use normalized verdict labels (`REAL`, `FAKE`, `UNCERTAIN`, `UNKNOWN`) across the entire pipeline.

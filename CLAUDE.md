# Multi-Agent Fake News Detection - Architecture & Guidelines

## 🏗️ Architecture Map
This project is structured into three main packages to separate concerns and isolate experimental features from production pipelines.

1. **`fake_news_detector/`** - 🚀 **Production Application**
   - The main, stable pipeline with UI (Streamlit).
   - Proven components only (RAG, basic baseline models).

2. **`shared_fact_checking/`** - 🧩 **Core Utilities**
   - Pure, stateless logic shared across modules.
   - Core retrieval policies, validation schemas, evaluation metrics.

3. **`trust_agents/`** - 🔬 **Experimental / Research**
   - Unstable, cutting-edge multi-agent orchestration.
   - Contains `agents2.0/` for experimental approaches like Delphi Jury and LoCal.
   - **Rule**: Do not import from here into `fake_news_detector` without rigorous testing.

## 🛠️ Development Guidelines
- **Linting & Formatting**: We use `ruff` with a line length of `120`. Run `uv run ruff check --fix src tests scripts` before committing.
- **Testing**: Maintain high test coverage. Avoid global configuration states (e.g., `set_llm_config`) that leak between tests.
- **Check-in**: Always run `scripts/quick_check.ps1` before pushing to ensure stability.

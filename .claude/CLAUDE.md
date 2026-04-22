# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Vietnamese Fake News Detection using **Multi-Agent RAG** (Retrieval Augmented Generation). The system uses 3 specialized agents with hybrid RAG (FAISS + Google Search) to verify claims and prevent hallucinations through citation validation.

---

## Architecture: 3-Agent Pipeline

```
[User Input]
    │
    ▼
┌─────────────────────────────────────────┐
│ Agent 1: Claim Extractor                │
│ src/agents/claim_extractor.py          │
│ - Extract claims from text              │
│ - Classify Fact vs Opinion              │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Agent 2: Evidence Retriever             │
│ src/agents/evidence_retriever.py       │
│ - Search FAISS (ViFactCheck dataset)   │
│ - Fallback to Google Search if weak     │
│ - Merge and deduplicate results         │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Agent 3: Reasoning (NVIDIA NIM)         │
│ src/agents/reasoning.py                 │
│ - Constraint prompting for JSON output │
│ - Citation validation                   │
│ - Streaming response for UI             │
└─────────────────────────────────────────┘
    │
    ▼
[Score + Label (REAL/FAKE/UNVERIFIABLE) + Evidence]
```

## Hybrid RAG Strategy

- **FAISS first**: Fast local search against ViFactCheck dataset
- **Confidence threshold**: If FAISS score < `settings.similarity_threshold`, trigger web search
- **Web search APIs**: Serper.dev or Tavily for real-time Google results
- **Merging**: Deduplicate by content similarity, sort by score

## Key Files

| File | Purpose |
|------|---------|
| `src/agents/reasoning.py` | Agent 3: LLM reasoning with citation validation |
| `src/rag/retriever.py` | Hybrid retrieval orchestration |
| `src/rag/vector_store.py` | FAISS index management |
| `src/utils/citation_checker.py` | Validates LLM citations exist in evidence |
| `src/trust_agents/orchestrator.py` | Main pipeline coordinator |
| `src/trust_agents/llm/groq_client.py` | Groq LLM API client |
| `notebooks/trust_agents_colab.ipynb` | Google Colab workflow (GPU) |

---

## Commands

```bash
# Install dependencies (uses uv, not poetry)
uv sync

# Download dataset and build FAISS index
uv run python scripts/download_data.py --build-index --max-samples 5000

# Run Streamlit app
uv run streamlit run src/fake_news_detector/app.py

# Run tests
uv run pytest tests/ -v

# Single test file
uv run pytest tests/unit/test_citation_checker.py -v

# Lint & format
uv run ruff check .
uv run ruff format .

# Type check
uv run mypy src/
```

## Environment Variables

```bash
GEMINI_API_KEY=xxxxx          # Google Gemini
GROQ_API_KEY=xxxxx           # Groq API (Free tier)
NVIDIA_API_KEY=nvapi-xxxxx     # NVIDIA NIM for LLM
SERPER_API_KEY=xxxxx          # Google Search
TAVILY_API_KEY=xxxxx          # Alternative search
HF_TOKEN=hf_xxxxx             # HuggingFace
```

## Tech Stack

- **LLM**: NVIDIA NIM (`google/gemma-4-31b-it`)
- **Baseline Model**: PhoBERT (`vinai/phobert-base`) + sliding window
- **RAG**: FAISS + Serper.dev/Tavily
- **UI**: Streamlit with streaming responses
- **Vietnamese NLP**: underthesea, sentence-transformers

## Rules

Read the rules in `.claude/rules/` for detailed guidelines:

- **code-style.md**: Code style and formatting (88 char line length, Ruff)
- **testing.md**: Testing conventions (AAA pattern, pytest fixtures)
- **security.md**: Security guidelines (OWASP Top 10)



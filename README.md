# Vietnamese Fake News Detection

Hệ thống phát hiện tin giả tiếng Việt sử dụng baseline PhoBERT kết hợp Multi-Agent RAG (TRUST Agents).

## Tính năng chính

- **TRUST Agents**: Pipeline đa tác tử gồm Claim Extraction, Evidence Retrieval, Verification, Explanation.
- **Hybrid RAG**: Kết hợp tìm kiếm vector nội bộ (FAISS) và tìm kiếm web (Serper/Tavily).
- **Stylistic Analysis**: Phân tích đặc điểm văn phong bài viết.
- **Baseline Model**: Phân tích bằng mô hình PhoBERT-base.
- **Terminal Runner**: Chạy baseline-only, TRUST-only, hoặc full analysis qua CLI.

## Cài đặt

### Yêu cầu

- Python >= 3.10
- `uv`
- API keys cho LLM provider và web search provider nếu chạy pipeline đầy đủ

### Cài dependencies

```bash
uv sync --all-extras
```

### Cấu hình môi trường

Tạo `.env` từ `.env.example`, rồi điền key cần dùng:

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

## Sử dụng

### Chạy runner tương tác

```bash
uv run python scripts/interactive_runner.py
```

### Chạy TRUST orchestrator trực tiếp

```bash
uv run python -m trust_agents.orchestrator --text "Nội dung cần kiểm tra" --top-k 5
```

### Chạy research orchestrator

```bash
uv run python -m trust_agents.orchestrator_research --text "Nội dung cần kiểm tra"
```

### Huấn luyện baseline model

```bash
uv run python scripts/train_baseline.py
```

### Tải ViFactCheck và build FAISS index

```bash
uv run python scripts/download_data.py --build-index --max-samples 5000
```

## Kiến trúc

```
src/
├── fake_news_detector/          # Core application services, models, retrieval
│   ├── application/             # Analysis and index services
│   ├── config.py                # Runtime settings
│   ├── data/                    # Dataset loading and preprocessing
│   ├── models/                  # PhoBERT baseline and stylistic features
│   ├── rag/                     # FAISS and web retrieval wrappers
│   └── utils/                   # Citation and utility helpers
├── shared_fact_checking/        # Shared retrieval policies/services
└── trust_agents/                # TRUST multi-agent pipeline
    ├── orchestrator.py          # Production orchestrator
    ├── orchestrator_research.py # Research orchestrator
    ├── agents/                  # Claim, retrieval, verifier, explainer agents
    └── llm/                     # LLM factory and provider helpers
```

## TRUST Pipeline

```
Article Text
  -> Claim Extractor
  -> Evidence Retriever
  -> Verifier
  -> Explainer
  -> Summary + claim-level verdicts
```

## Testing

```bash
uv run pytest tests/ -v
uv run pytest tests/unit/test_citation_checker.py -v
uv run pytest tests/integration/ -v
```

## Quality checks

```bash
uv run scripts/smoke_test.py
uv run ruff check .
uv run ruff format .
uv run mypy src/
```

## Troubleshooting

### FAISS index missing

```bash
uv run python scripts/download_data.py --build-index --max-samples 5000
```

### HuggingFace model download failed

Check `HF_TOKEN` in `.env`, then retry command that loads model.

### LLM provider errors

Check selected `LLM_PROVIDER` and matching API key in `.env`.

## License

MIT

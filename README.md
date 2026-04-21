# Vietnamese Fake News Detection

Hệ thống **Multi-Agent RAG** phát hiện tin giả tiếng Việt, sử dụng PhoBERT + TRUST Agents (Multi-Agent) + NVIDIA NIM/Gemini.

## Mục lục

- [Tổng quan](#tổng-quan)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt nhanh](#cài-đặt-nhanh)
- [Cài đặt chi tiết](#cài-đặt-chi-tiết)
- [Cấu hình API Keys](#cấu-hình-api-keys)
- [Dataset và FAISS Index](#dataset-và-faiss-index)
- [Chạy ứng dụng](#chạy-ứng-dụng)
- [Kiến trúc hệ thống](#kiến-trúc-hệ-thống)
- [Testing](#testing)
- [Xử lý lỗi thường gặp](#xử-lý-lỗi-thường-gặp)

---

## Tổng quan

### Features chính

| Tính năng | Mô tả |
|-----------|-------|
| **TRUST Multi-Agent** | 4 agents chuyên biệt: Claim Extractor → Evidence Retriever → Verifier → Explainer |
| **Hybrid RAG** | Kết hợp FAISS vector search + Google Search fallback |
| **Vietnamese NLP** | Hỗ trợ xử lý tiếng Việt với underthesea, spaCy (multilingual) |
| **Citation Validation** | Ngăn chặn hallucination bằng kiểm tra trích dẫn |
| **Streaming UI** | Giao diện Streamlit real-time với pipeline selection |
| **Multi-LLM Support** | Hỗ trợ NVIDIA NIM, OpenAI, Google Gemini |

### Tech Stack

```
LLM Backend:     NVIDIA NIM (google/gemma-4-31b-it) / Gemini / OpenAI
Baseline Model:  PhoBERT (vinai/phobert-base)
Embeddings:      sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
Vector Store:    FAISS (ViFactCheck dataset)
Search APIs:      Serper.dev / Tavily
UI:              Streamlit
NLP:             underthesea, spaCy (xx_ent_wiki_sm)
Package Manager:  uv
```

---

## Yêu cầu hệ thống

### Prerequisites

- **Python**: 3.10 trở lên
- ** uv**: Package manager (hướng dẫn cài đặt bên dưới)
- **API Keys**: NVIDIA NIM, Serper.dev/Tavily, HuggingFace
- **RAM**: Tối thiểu 8GB (khuyến nghị 16GB)
- **Disk**: ~5GB cho models và dataset

### Kiểm tra Python version

```bash
python --version
# Cần output: Python 3.10.x hoặc cao hơn
```

---

## Cài đặt nhanh

```bash
# 1. Clone project (hoặc cd vào thư mục hiện tại)
cd D:\Work\project\mutiAgentfakeNew

# 2. Cài đặt uv nếu chưa có
pip install uv

# 3. Cài đặt dependencies
uv sync

# 4. Copy và cấu hình .env
copy .env.example .env
# Edit .env với API keys của bạn

# 5. Download dataset và build FAISS index
uv run python scripts/download_data.py --build-index --max-samples 5000

# 6. Chạy ứng dụng
uv run streamlit run src/fake_news_detector/app.py
```

Truy cập: **http://localhost:8501**

---

## Cài đặt chi tiết

### Bước 1: Cài đặt uv

`uv` là fast Python package manager được recommend cho project này.

```bash
# Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex

# Hoặc dùng pip
pip install uv

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Bước 2: Cài đặt Dependencies

```bash
# Từ thư mục project
uv sync

# Với dev dependencies (testing, linting)
uv sync --all-extras
```

### Bước 3: Cấu hình Environment Variables

```bash
# Tạo file .env từ template
copy .env.example .env
```

Mở file `.env` và điền các API keys cần thiết (xem phần [Cấu hình API Keys](#cấu-hình-api-keys)).

---

## Cấu hình API Keys

### NVIDIA NIM API (Bắt buộc)

Truy cập: https://www.nvidia.com/cdp

1. Đăng ký tài khoản NVIDIA Developer
2. Truy cập NVIDIA NIM API
3. Tạo API Key mới
4. Copy vào `.env`:

```bash
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxxxxxx
```

### Serper.dev - Google Search (Khuyến nghị)

Truy cập: https://serper.dev

1. Đăng ký tài khoản (có free tier: 2,500 queries/tháng)
2. Lấy API Key từ dashboard
3. Copy vào `.env`:

```bash
SERPER_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxx
```

### Tavily (Thay thế Serper)

Truy cập: https://tavily.com

```bash
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxxxxxxxxxx
```

### HuggingFace Token (Bắt buộc)

Truy cập: https://huggingface.co/settings/tokens

1. Tạo Access Token (read permission)
2. Copy vào `.env`:

```bash
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### File `.env` hoàn chỉnh

```bash
# NVIDIA NIM API (for google/gemma-4-31b-it)
NVIDIA_API_KEY=nvapi-xxxxx

# Serper.dev (Google Search API)
SERPER_API_KEY=xxxxx

# Tavily (alternative to Serper)
TAVILY_API_KEY=xxxxx

# HuggingFace (for PhoBERT model download)
HF_TOKEN=hf_xxxxx

# Optional: For model caching
HF_HOME=./models_cache
TRANSFORMERS_CACHE=./models_cache/transformers
```

---

## Dataset và FAISS Index

### ViFactCheck Dataset

Project sử dụng dataset [ViFactCheck](https://huggingface.co/datasets/tranthaihoa/vifactcheck) từ HuggingFace.

### Download và Build Index

```bash
# Download dataset + build FAISS index (5000 samples)
uv run python scripts/download_data.py --build-index --max-samples 5000

# Hoặc với custom output directory
uv run python scripts/download_data.py --build-index --max-samples 5000 --output-dir ./data

# Với nhiều samples hơn (tốt hơn nhưng chậm hơn)
uv run python scripts/download_data.py --build-index --max-samples 10000
```

### Kiểm tra Index đã được tạo

```bash
# Kiểm tra thư mục FAISS index
ls -la data/faiss_index/

# Output cần có các file: index.faiss, metadata.json
```

### Tải models cần thiết

Models sẽ được tự động tải khi chạy ứng dụng lần đầu:

| Model | Size | Mục đích |
|-------|------|----------|
| `vinai/phobert-base` | ~250MB | Vietnamese NLP baseline |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | ~500MB | Embeddings |
| `xx_ent_wiki_sm` (spaCy) | ~50MB | Multilingual NER |

```bash
# Pre-download models (tùy chọn, giúp chạy nhanh hơn lần đầu)
uv run python -c "
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

print('Downloading PhoBERT...')
AutoTokenizer.from_pretrained('vinai/phobert-base')
AutoModel.from_pretrained('vinai/phobert-base')

print('Downloading Sentence Transformer...')
SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

print('Done!')
"
```

---

## Chạy ứng dụng

### Khởi động Streamlit

```bash
# Từ thư mục project
uv run streamlit run src/fake_news_detector/app.py
```

Browser sẽ tự động mở tại **http://localhost:8501**

### Giao diện và cách sử dụng

#### Sidebar - Cài đặt

```
Pipeline: Chọn "TRUST Multi-Agent" hoặc "Legacy"
├── TRUST Multi-Agent: Sử dụng hệ thống 4 agents mới (recommended)
└── Legacy: Hệ thống cũ 3 agents
```

#### Input

1. **Nhập nội dung tin tức** vào text area
2. Click **🔍 Phân tích** để bắt đầu

#### Kết quả

| Section | Mô tả |
|---------|-------|
| **Baseline Results** | Real/Fake probability từ PhoBERT |
| **TRUST Agents** | Chi tiết claims với verdicts (✅ TRUE / ❌ FALSE / ❓ UNCERTAIN) |
| **Stylistic Features** | Đặc điểm văn phong (tỷ lệ HOA, cảm xúc, giật gân) |
| **Evidence** | Evidence passages được retrieve |

### Chạy CLI

```bash
# Fact-check trực tiếp từ command line
uv run python -m trust_agents.orchestrator --text "Nội dung tin cần kiểm tra" --top-k 5

# Với output file
uv run python -m trust_agents.orchestrator --text "Tin cần kiểm tra" --output result.json
```

---

## Kiến trúc hệ thống

### TRUST Multi-Agent Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│                     INPUT: Article Text                       │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│              Agent 1: Claim Extractor                        │
│  ├── NER-based extraction (spaCy multilingual)             │
│  ├── Dependency parsing (subject-verb-object)              │
│  └── LLM zero-shot reasoning                                │
│  Output: List of factual claims                            │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│              Agent 2: Evidence Retriever                      │
│  ├── FAISS vector search (ViFactCheck)                       │
│  ├── Web search fallback (Serper/Tavily)                     │
│  └── Merge & deduplicate results                             │
│  Output: Top-K evidence passages                            │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│              Agent 3: Verifier                               │
│  ├── Compare claim vs evidence                               │
│  ├── Verdict: TRUE / FALSE / UNCERTAIN                     │
│  └── Confidence scoring                                     │
│  Output: Verdict + reasoning + confidence                   │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│              Agent 4: Explainer                              │
│  ├── Generate human-readable explanation                   │
│  ├── Summary statistics                                      │
│  └── Citation formatting                                    │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│              OUTPUT: Complete Analysis                       │
│  ├── Claims found                                            │
│  ├── Verdicts (TRUE/FALSE/UNCERTAIN)                        │
│  ├── Evidence citations                                     │
│  └── Summary statistics                                     │
└──────────────────────────────────────────────────────────────┘
```

### Hybrid RAG Strategy

```
Claim → FAISS Search
            │
            ├─ Score ≥ 0.5 → Use evidence
            │
            └─ Score < 0.5 → Web Search Fallback
                              │
                              └─ Merge & Rank
```

### Project Structure

```
src/
├── fake_news_detector/          # Main application
│   ├── config.py                # Settings & Pydantic config
│   ├── app.py                   # Streamlit UI
│   ├── data/
│   │   ├── loader.py            # ViFactCheck dataset loader
│   │   └── preprocessing.py     # Vietnamese NLP preprocessing
│   ├── models/
│   │   ├── baseline.py         # PhoBERT sliding window
│   │   └── stylistic.py        # Feature extraction
│   ├── rag/
│   │   ├── vector_store.py     # FAISS index management
│   │   ├── retriever.py        # Hybrid retrieval orchestration
│   │   └── web_search.py       # Serper/Tavily APIs
│   └── visualization/
│       └── wordcloud.py        # Word analysis
│
└── trust_agents/               # Multi-agent system
    ├── orchestrator.py         # TRUST pipeline coordinator
    └── agents/
        ├── claim_extractor.py  # Agent 1 (sync wrapper)
        ├── claim_extractor_tools.py  # Tools: NER, Dependency, LLM
        ├── evidence_retrieval.py       # Agent 2 (sync wrapper)
        ├── retrieval_agent_tools.py   # Tools: FAISS, Web Search
        ├── verifier.py          # Agent 3 (sync wrapper)
        └── explainer.py        # Agent 4 (sync wrapper)
```

### Configuration Reference

| Setting | Mặc định | Mô tả |
|---------|----------|-------|
| `phobert_model` | `vinai/phobert-base` | PhoBERT model |
| `embedding_model` | `paraphrase-multilingual-MiniLM-L12-v2` | Sentence transformer |
| `llm_model` | `google/gemma-4-31b-it` | LLM via NVIDIA NIM |
| `faiss_index_path` | `./data/faiss_index` | FAISS index location |
| `similarity_threshold` | `0.5` | Confidence threshold for web search |
| `top_k_evidence` | `5` | Evidence passages per claim |

---

## Testing

### Chạy tất cả tests

```bash
uv run pytest tests/ -v
```

### Chạy specific test file

```bash
uv run pytest tests/unit/test_citation_checker.py -v
```

### Chạy với coverage

```bash
uv run pytest --cov=src --cov-report=term-missing tests/
```

### Markers

```bash
# Unit tests only
uv run pytest -m unit

# Integration tests only
uv run pytest -m integration

# Skip slow tests
uv run pytest -m "not slow"
```

### Linting & Type Checking

```bash
# Ruff check
uv run ruff check src/

# Auto-fix issues
uv run ruff check --fix src/

# Format code
uv run ruff format src/

# Type check
uv run mypy src/
```

---

## Xử lý lỗi thường gặp

### Lỗi: `ModuleNotFoundError`

```bash
# Thiếu dependencies → Re-install
uv sync
```

### Lỗi: `HF_TOKEN` hoặc model download failed

```bash
# Kiểm tra HuggingFace token
# Ensure đã add vào .env: HF_TOKEN=hf_xxxxx

# Thử login HuggingFace
uv run huggingface-cli login
```

### Lỗi: `NVIDIA_API_KEY` invalid

```bash
# Verify API key
curl https://integrate.api.nvidia.com/v1/models \
  -H "Authorization: Bearer $NVIDIA_API_KEY"

# Kiểm tra quota còn không
```

### Lỗi: FAISS index not found

```bash
# Rebuild index
uv run python scripts/download_data.py --build-index --max-samples 5000

# Verify path trong config
# Kiểm tra data/faiss_index/ có tồn tại không
```

### Lỗi: spaCy model missing

```bash
# Download multilingual model
uv run python -m spacy download xx_ent_wiki_sm

# Hoặc English model
uv run python -m spacy download en_core_web_sm
```

### Lỗi: Streamlit port already in use

```bash
# Dùng port khác
uv run streamlit run src/fake_news_detector/app.py --server.port 8502

# Hoặc kill process đang dùng port 8501
# Windows:
netstat -ano | findstr :8501
taskkill /PID <PID> /F
```

### Lỗi: underthesea not working

```bash
# Cài đặt underthesea
uv pip install underthesea

# Nếu vẫn lỗi, kiểm tra PyTorch
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Performance Tips

| Vấn đề | Giải pháp |
|--------|-----------|
| Chạy chậm lần đầu | Pre-download models (xem phần Dataset) |
| RAM cao | Giảm `--max-samples` khi build index |
| FAISS search chậm | Giảm `top_k_evidence` trong config |

---

## License

MIT

"""Application configuration using Python standard dataclasses."""
import os
from dataclasses import dataclass, field
from typing import Literal

from dotenv import load_dotenv

# Explicitly load .env on module import to emulate Pydantic BaseSettings behavior
load_dotenv()

@dataclass
class Settings:
    """Application settings loaded from environment variables."""

    # NVIDIA NIM API
    nvidia_api_key: str = field(default_factory=lambda: os.getenv("NVIDIA_API_KEY", ""))

    # Search APIs
    serper_api_key: str = field(default_factory=lambda: os.getenv("SERPER_API_KEY", ""))
    tavily_api_key: str = field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))

    # HuggingFace
    hf_token: str = field(default_factory=lambda: os.getenv("HF_TOKEN", ""))

    # Model settings
    phobert_model: str = field(default_factory=lambda: os.getenv("PHOBERT_MODEL", "vinai/phobert-base"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "openai/gpt-oss-120b"))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"))

    # RAG settings
    faiss_index_path: str = field(default_factory=lambda: os.getenv("FAISS_INDEX_PATH", "./data/faiss_index"))
    top_k_faiss: int = field(default_factory=lambda: int(os.getenv("TOP_K_FAISS", "5")))
    top_k_google: int = field(default_factory=lambda: int(os.getenv("TOP_K_GOOGLE", "3")))
    similarity_threshold: float = field(default_factory=lambda: float(os.getenv("SIMILARITY_THRESHOLD", "0.7")))

    # Search engine choice
    search_provider: Literal["serper", "tavily"] = field(default_factory=lambda: os.getenv("SEARCH_PROVIDER", "serper")) # type: ignore

    # Application settings
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "False").lower() == "true")
    max_tokens: int = field(default_factory=lambda: int(os.getenv("MAX_TOKENS", "2048")))
    streaming: bool = field(default_factory=lambda: os.getenv("STREAMING", "True").lower() == "true")

    # Trusted Vietnamese news sources for consistency check
    trusted_sources: list[str] = field(default_factory=lambda: [
        "vnexpress.net",
        "nhandan.vn",
        "baotintuc.vn",
        "thanhnien.vn",
        "tuoitre.vn",
    ])

    # Cache settings
    cache_ttl: int = field(default_factory=lambda: int(os.getenv("CACHE_TTL", "3600")))


settings = Settings()

"""Application configuration with Pydantic Settings."""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Allow extra env vars like HF_HOME, TRANSFORMERS_CACHE
    )

    # NVIDIA NIM API
    nvidia_api_key: str = ""

    # Search APIs
    serper_api_key: str = ""
    tavily_api_key: str = ""

    # HuggingFace
    hf_token: str = ""

    # Model settings
    phobert_model: str = "vinai/phobert-base"
    llm_model: str = "openai/gpt-oss-120b"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # RAG settings
    faiss_index_path: str = "./data/faiss_index"
    top_k_faiss: int = 5
    top_k_google: int = 3
    similarity_threshold: float = 0.7

    # Search engine choice
    search_provider: Literal["serper", "tavily"] = "serper"

    # Application settings
    debug: bool = False
    max_tokens: int = 2048
    streaming: bool = True

    # Trusted Vietnamese news sources for consistency check
    trusted_sources: list[str] = [
        "vnexpress.net",
        "nhandan.vn",
        "baotintuc.vn",
        "thanhnien.vn",
        "tuoitre.vn",
    ]

    # Cache settings
    cache_ttl: int = 3600


settings = Settings()
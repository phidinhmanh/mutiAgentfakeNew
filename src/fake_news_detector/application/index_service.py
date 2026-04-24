"""Application service for sample data and vector index flows."""
from __future__ import annotations

from fake_news_detector.config import settings
from fake_news_detector.data.loader import load_vifactcheck
from fake_news_detector.rag.vector_store import get_vector_store


def load_sample_claim_and_evidence() -> tuple[str, str]:
    """Load the first ViFactCheck sample for UI preview."""
    dataset = load_vifactcheck("train")
    sample = dataset[0]
    return sample.get("claim", ""), sample.get("evidence", "")


def build_vector_index(max_docs: int = 1000) -> None:
    """Build and persist the FAISS vector index from dataset evidence."""
    dataset = load_vifactcheck("train")
    vector_store = get_vector_store()
    docs = [
        {"content": item.get("evidence", ""), "label": item.get("label", "")}
        for item in dataset
        if item.get("evidence")
    ]
    vector_store.add_documents(docs[:max_docs])
    vector_store.save(settings.faiss_index_path)

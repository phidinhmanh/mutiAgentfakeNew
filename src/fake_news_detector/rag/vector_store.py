"""FAISS vector store for semantic search."""
import logging
import os
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

from fake_news_detector.config import settings
from fake_news_detector.data.preprocessing import preprocess_for_embedding

logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS-based vector store for evidence retrieval."""

    def __init__(self, embed_model: str | None = None) -> None:
        """Initialize vector store.

        Args:
            embed_model: Sentence transformer model name
        """
        self.embed_model_name = embed_model or settings.embedding_model
        self.embedder = SentenceTransformer(self.embed_model_name)
        self.index: faiss.IndexFlatIP | None = None
        self.documents: list[dict[str, Any]] = []
        self.embedding_dim: int = self.embedder.get_sentence_embedding_dimension()

    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        """Add documents to the vector store.

        Args:
            documents: List of documents with 'content' and metadata
        """
        if not documents:
            return

        texts = [preprocess_for_embedding(doc.get("content", "")) for doc in documents]
        embeddings = self.embedder.encode(texts, show_progress_bar=True)

        if self.index is None:
            self.index = faiss.IndexFlatIP(self.embedding_dim)

        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype(np.float32))
        self.documents.extend(documents)

        logger.info(f"Added {len(documents)} documents, total: {self.index.ntotal}")

    def similarity_search(
        self, query: str, k: int = 5
    ) -> list[dict[str, Any]]:
        """Search for similar documents.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of similar documents with scores
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Vector store is empty")
            return []

        query_embedding = self.embedder.encode([preprocess_for_embedding(query)])
        faiss.normalize_L2(query_embedding)

        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(
            query_embedding.astype(np.float32), k
        )

        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc["score"] = float(score)
                doc["rank"] = i + 1
                results.append(doc)

        return results

    def save(self, path: str | Path) -> None:
        """Save vector store to disk.

        Args:
            path: Path to save the index
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.index is not None:
            index_path = path / "index.faiss"
            faiss.write_index(self.index, str(index_path))

        docs_path = path / "documents.pkl"
        with open(docs_path, "wb") as f:
            pickle.dump(self.documents, f)

        logger.info(f"Saved vector store to {path}")

    def load(self, path: str | Path) -> None:
        """Load vector store from disk.

        Args:
            path: Path to load the index from
        """
        path = Path(path)

        index_path = path / "index.faiss"
        if index_path.exists():
            self.index = faiss.read_index(str(index_path))
            logger.info(f"Loaded index with {self.index.ntotal} vectors")

        docs_path = path / "documents.pkl"
        if docs_path.exists():
            with open(docs_path, "rb") as f:
                self.documents = pickle.load(f)
            logger.info(f"Loaded {len(self.documents)} documents")


_vector_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """Get singleton vector store instance.

    Returns:
        VectorStore instance
    """
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
        index_path = Path(settings.faiss_index_path)
        if index_path.exists():
            _vector_store.load(index_path)
    return _vector_store
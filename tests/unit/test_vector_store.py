"""Tests for vector store (FAISS)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

from fake_news_detector.rag.vector_store import VectorStore, get_vector_store


class TestVectorStoreInit:
    """Test VectorStore initialization."""

    def test_init_creates_embedder(self, mock_sentence_transformer: Mock) -> None:
        """VectorStore initializes with embedder."""
        vs = VectorStore()
        assert vs.embedder is not None

    def test_init_empty_index(self, mock_sentence_transformer: Mock) -> None:
        """VectorStore starts with empty index."""
        vs = VectorStore()
        assert vs.index is None
        assert vs.documents == []


class TestVectorStoreAddDocuments:
    """Test VectorStore.add_documents method."""

    def test_add_documents_empty_list(self, mock_sentence_transformer: Mock) -> None:
        """Adding empty list does nothing."""
        vs = VectorStore()
        vs.add_documents([])
        assert vs.index is None

    def test_add_documents_single(self, mock_sentence_transformer: Mock) -> None:
        """Single document is added correctly."""
        vs = VectorStore()
        documents = [{"content": "Test content", "id": "1"}]
        vs.add_documents(documents)
        assert vs.index is not None
        assert len(vs.documents) == 1

    def test_add_documents_multiple(self, mock_sentence_transformer: Mock) -> None:
        """Multiple documents are added correctly."""
        vs = VectorStore()
        documents = [
            {"content": "First content", "id": "1"},
            {"content": "Second content", "id": "2"},
        ]
        vs.add_documents(documents)
        assert len(vs.documents) == 2

    def test_add_documents_twice(self, mock_sentence_transformer: Mock) -> None:
        """Documents can be added in batches."""
        vs = VectorStore()
        vs.add_documents([{"content": "First", "id": "1"}])
        vs.add_documents([{"content": "Second", "id": "2"}])
        assert len(vs.documents) == 2


class TestVectorStoreSimilaritySearch:
    """Test VectorStore.similarity_search method."""

    def test_similarity_search_empty_index(
        self, mock_sentence_transformer: Mock
    ) -> None:
        """Empty index returns empty list."""
        vs = VectorStore()
        results = vs.similarity_search("test query")
        assert results == []

    def test_similarity_search_single_result(
        self, mock_sentence_transformer: Mock
    ) -> None:
        """Search returns results with scores."""
        vs = VectorStore()
        vs.add_documents([{"content": "Test content", "id": "1"}])
        results = vs.similarity_search("test query", k=1)
        assert len(results) == 1
        assert "score" in results[0]
        assert "content" in results[0]

    def test_similarity_search_k_limits_results(
        self, mock_sentence_transformer: Mock
    ) -> None:
        """k parameter limits results."""
        vs = VectorStore()
        vs.add_documents([{"content": f"Content {i}", "id": str(i)} for i in range(5)])
        results = vs.similarity_search("test", k=2)
        assert len(results) <= 2

    def test_similarity_search_includes_rank(
        self, mock_sentence_transformer: Mock
    ) -> None:
        """Results include rank field."""
        vs = VectorStore()
        vs.add_documents(
            [
                {"content": "First", "id": "1"},
                {"content": "Second", "id": "2"},
            ]
        )
        results = vs.similarity_search("test", k=2)
        assert all("rank" in r for r in results)


class TestVectorStoreSaveLoad:
    """Test VectorStore save/load methods."""

    def test_save_creates_directory(
        self, mock_sentence_transformer: Mock, tmp_path: Path
    ) -> None:
        """Save creates parent directories."""
        vs = VectorStore()
        vs.add_documents([{"content": "Test", "id": "1"}])
        save_path = tmp_path / "nested" / "dir" / "store"
        vs.save(save_path)
        assert save_path.exists()

    def test_load_reads_existing_index(
        self, mock_sentence_transformer: Mock, tmp_path: Path
    ) -> None:
        """Load reads from disk."""
        vs = VectorStore()
        vs.add_documents([{"content": "Test content", "id": "1"}])
        save_path = tmp_path / "store"
        vs.save(save_path)

        new_vs = VectorStore()
        new_vs.load(save_path)
        assert len(new_vs.documents) == 1

    def test_load_nonexistent_path(
        self, mock_sentence_transformer: Mock, tmp_path: Path
    ) -> None:
        """Load from nonexistent path does nothing."""
        vs = VectorStore()
        vs.load(tmp_path / "nonexistent")
        assert vs.index is None
        assert vs.documents == []


class TestGetVectorStore:
    """Test get_vector_store singleton function."""

    @patch("fake_news_detector.rag.vector_store.VectorStore")
    @patch("fake_news_detector.rag.vector_store._vector_store", None)
    @patch("fake_news_detector.rag.vector_store.settings")
    def test_returns_vector_store_instance(
        self, mock_settings: Mock, mock_vs_class: Mock
    ) -> None:
        """Returns VectorStore instance."""
        mock_settings.faiss_index_path = "/tmp/nonexistent"
        mock_settings.embedding_model = "mock-model"
        mock_vs_instance = Mock()
        mock_vs_class.return_value = mock_vs_instance

        get_vector_store()
        mock_vs_class.assert_called_once()

    @patch("fake_news_detector.rag.vector_store._vector_store")
    def test_singleton_returns_same_instance(self, mock_vs: Mock) -> None:
        """Called multiple times returns same instance."""
        result = get_vector_store()
        assert result is mock_vs

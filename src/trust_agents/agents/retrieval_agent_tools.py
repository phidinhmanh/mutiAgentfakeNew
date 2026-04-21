# -*- coding: utf-8 -*-

"""
Retrieval Agent Tools - Tools for evidence retrieval using hybrid search.

Tools used by the Evidence Retriever ReAct Agent:
- Search Tool: Hybrid FAISS + keyword retrieval
- Index Tool: Build or load document index
- Get Passage Tool: Retrieve specific passage by ID

Note: These tools are designed to integrate with our RAG system.
Supports OpenAI, Google Gemini (AI Studio), and NVIDIA NIM backends.
"""

import os
import json
import logging
from typing import Any, Optional

from langchain_core.tools import tool

logger = logging.getLogger("TRUST_agents.agents.retrieval_agent_tools")
logger.propagate = True


def _get_rag_components():
    """Get RAG components from the existing project.

    Returns:
        Tuple of (vector_store_fn, retrieve_evidence_function)
    """
    try:
        from fake_news_detector.rag.vector_store import get_vector_store
        from fake_news_detector.rag.retriever import retrieve_evidence
        return get_vector_store, retrieve_evidence
    except ImportError as e:
        logger.warning(f"Could not import RAG components: {e}")
        return None, None


@tool()
async def search_evidence_tool(query: str, top_k: int = 5, use_web_search: bool = True) -> str:
    """
    Search for evidence passages relevant to a query using hybrid retrieval.

    Uses FAISS vector search first, then falls back to web search if confidence
    is below threshold.

    Args:
        query: The search query or claim to find evidence for
        top_k: Number of top results to return (default: 5)
        use_web_search: Whether to use web search as fallback (default: True)

    Returns:
        JSON string with evidence passages
    """
    logger.info(f"[DEBUG] search_evidence_tool called: query='{query[:50]}...', top_k={top_k}")

    try:
        vector_store_fn, retrieve_fn = _get_rag_components()

        if vector_store_fn is not None:
            # Use actual RAG system
            vector_store = vector_store_fn()
            faiss_results = vector_store.similarity_search(query, k=top_k)

            logger.info(f"FAISS returned {len(faiss_results)} results")

            # Calculate confidence
            if faiss_results:
                avg_score = sum(r.get("score", 0) for r in faiss_results) / len(faiss_results)
                max_score = max(r.get("score", 0) for r in faiss_results)
                confidence = (avg_score + max_score) / 2
            else:
                confidence = 0.0

            # Use web search fallback if needed
            if use_web_search and confidence < 0.5:
                logger.info(f"Low confidence ({confidence:.3f}), adding web search...")
                from fake_news_detector.rag.web_search import search_web
                web_results = search_web(query, num_results=top_k)
                logger.info(f"Web search returned {len(web_results)} results")
                results = _merge_faiss_web(faiss_results, web_results, max_results=top_k)
            else:
                results = faiss_results[:top_k]

            evidence = []
            for i, r in enumerate(results):
                evidence.append({
                    "id": r.get("id", f"passage_{i+1:03d}"),
                    "text": r.get("content", r.get("text", "")),
                    "source": r.get("source", "ViFactCheck"),
                    "score": r.get("score", 0.5),
                    "url": r.get("url", ""),
                    "metadata": r.get("metadata", {})
                })
        else:
            # Mock results when RAG not available
            logger.warning("RAG not available, using mock evidence")
            evidence = [
                {
                    "id": f"mock_{i+1:03d}",
                    "text": f"Mock evidence for '{query[:30]}...'. This is a placeholder.",
                    "source": "Mock Dataset",
                    "score": 0.8 - (i * 0.1)
                }
                for i in range(min(top_k, 3))
            ]

        result = {
            "evidence": evidence,
            "query": query,
            "top_k": top_k,
            "total_found": len(evidence),
            "avg_score": sum(e.get("score", 0) for e in evidence) / len(evidence) if evidence else 0
        }

        logger.info(f"search_evidence_tool completed: {len(evidence)} results")
        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Error in search_evidence_tool: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return json.dumps({
            "evidence": [],
            "error": str(e),
            "query": query,
            "top_k": top_k
        }, ensure_ascii=False)


def _merge_faiss_web(
    faiss_results: list[dict[str, Any]],
    web_results: list[dict[str, Any]],
    max_results: int = 8
) -> list[dict[str, Any]]:
    """Merge FAISS and web search results.

    Args:
        faiss_results: Results from FAISS
        web_results: Results from web search
        max_results: Maximum number of results to return

    Returns:
        Merged and sorted results
    """
    seen_contents = set()
    merged = []

    for result in faiss_results:
        content = result.get("content", result.get("text", ""))[:200]
        if content not in seen_contents:
            seen_contents.add(content)
            result["source"] = result.get("source", "vi_fact_check")
            merged.append(result)

    for result in web_results:
        content = result.get("content", result.get("text", ""))[:200]
        if content not in seen_contents:
            seen_contents.add(content)
            result["source"] = result.get("source", "web_search")
            merged.append(result)

    merged.sort(key=lambda x: x.get("score", 0), reverse=True)
    return merged[:max_results]


@tool()
async def get_passage_tool(passage_id: str) -> str:
    """
    Get details of a specific passage by ID.

    Args:
        passage_id: ID of the passage to retrieve

    Returns:
        JSON string with passage details
    """
    logger.info(f"[DEBUG] get_passage_tool called: passage_id={passage_id}")

    try:
        vector_store_fn, _ = _get_rag_components()

        if vector_store_fn is not None:
            vector_store = vector_store_fn()
            # Search for passage by iterating documents
            passage = None
            for i, doc in enumerate(vector_store.documents):
                if doc.get("id") == passage_id or i == int(passage_id.split("_")[-1]) - 1:
                    passage = doc
                    passage["id"] = passage_id
                    break

            if passage is None:
                passage = {
                    "id": passage_id,
                    "text": f"Passage {passage_id} not found in index.",
                    "source": "Unknown",
                    "metadata": {}
                }
        else:
            passage = {
                "id": passage_id,
                "text": f"Text for passage {passage_id}. Load actual passage from FAISS index.",
                "source": "Unknown",
                "metadata": {}
            }

        logger.info(f"get_passage_tool completed")
        return json.dumps(passage, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Error in get_passage_tool: {e}")
        return json.dumps({
            "id": passage_id,
            "error": str(e)
        }, ensure_ascii=False)


@tool()
async def list_indexed_documents_tool() -> str:
    """
    List all indexed documents in the evidence corpus.

    Returns:
        JSON string with list of indexed documents
    """
    logger.info(f"[DEBUG] list_indexed_documents_tool called")

    try:
        vector_store_fn, _ = _get_rag_components()

        if vector_store_fn is not None:
            vector_store = vector_store_fn()
            docs = []
            seen_ids = set()
            for i, doc in enumerate(vector_store.documents):
                doc_id = doc.get("id", f"doc_{i:03d}")
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    docs.append({
                        "id": doc_id,
                        "title": doc.get("title", doc.get("content", "")[:100]),
                        "source": doc.get("source", "ViFactCheck"),
                        "url": doc.get("url", "")
                    })
        else:
            docs = [
                {"id": "doc_001", "title": "ViFactCheck Dataset", "source": "Vietnamese Fact-Check", "passage_count": 5000}
            ]

        result = {
            "documents": docs,
            "total_documents": len(docs)
        }

        logger.info(f"list_indexed_documents_tool completed: {len(docs)} documents")
        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Error in list_indexed_documents_tool: {e}")
        return json.dumps({
            "documents": [],
            "error": str(e)
        }, ensure_ascii=False)


@tool()
async def index_documents_tool(corpus_path: str, chunk_size: int = 160) -> str:
    """
    Index documents from a corpus for retrieval.

    Args:
        corpus_path: Path to directory or file to index
        chunk_size: Maximum words per chunk

    Returns:
        JSON string with indexing statistics
    """
    logger.info(f"[DEBUG] index_documents_tool called: path={corpus_path}, chunk_size={chunk_size}")

    try:
        vector_store_fn, _ = _get_rag_components()

        if vector_store_fn is not None:
            # Try to import the indexing functionality
            try:
                from fake_news_detector.data.preprocessing import preprocess_for_embedding
                from fake_news_detector.config import settings
                import glob
                import os

                documents = []
                if os.path.isdir(corpus_path):
                    for filepath in glob.glob(os.path.join(corpus_path, "**/*.txt"), recursive=True):
                        with open(filepath, "r", encoding="utf-8") as f:
                            content = f.read()
                            documents.append({
                                "content": content,
                                "source": filepath,
                                "id": os.path.basename(filepath)
                            })
                elif os.path.isfile(corpus_path):
                    with open(corpus_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        documents.append({
                            "content": content,
                            "source": corpus_path,
                            "id": os.path.basename(corpus_path)
                        })

                vector_store = vector_store_fn()
                vector_store.add_documents(documents)
                stats = {
                    "success": True,
                    "documents_indexed": len(documents),
                    "chunks_created": len(documents)
                }
            except Exception as ie:
                logger.warning(f"Could not index documents: {ie}")
                stats = {
                    "success": False,
                    "documents_indexed": 0,
                    "chunks_created": 0,
                    "message": f"Indexing not available: {str(ie)}"
                }
        else:
            stats = {
                "success": True,
                "documents_indexed": 0,
                "chunks_created": 0,
                "message": "Mock indexing - RAG not available"
            }

        result = {
            "success": True,
            "corpus_path": corpus_path,
            "chunk_size": chunk_size,
            **stats
        }

        logger.info(f"index_documents_tool completed: {stats.get('documents_indexed', 0)} documents indexed")
        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Error in index_documents_tool: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "corpus_path": corpus_path
        }, ensure_ascii=False)
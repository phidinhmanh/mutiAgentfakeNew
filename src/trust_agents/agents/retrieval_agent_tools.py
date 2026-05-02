"""
Retrieval Agent Tools - Tools for evidence retrieval using hybrid search.

Tools used by the Evidence Retriever ReAct Agent:
- Search Tool: Hybrid FAISS + keyword retrieval
- Index Tool: Build or load document index
- Get Passage Tool: Retrieve specific passage by ID

Note: These tools are designed to integrate with our RAG system.
Supports OpenAI, Google Gemini (AI Studio), and NVIDIA NIM backends.
"""

import json
import logging
from typing import Any

from langchain_core.tools import tool

from trust_agents.config import settings
from shared_fact_checking.retrieval.policy import merge_results
from shared_fact_checking.retrieval.service import retrieve_with_fallback

logger = logging.getLogger("TRUST_agents.agents.retrieval_agent_tools")
logger.propagate = True


def _get_rag_components():
    """Get RAG components from the existing project.

    Returns:
        Tuple of (vector_store_fn, retrieve_evidence_function)
    """
    try:
        from trust_agents.rag.retriever import retrieve_evidence
        from trust_agents.rag.vector_store import get_vector_store

        return get_vector_store, retrieve_evidence
    except ImportError as e:
        logger.warning(f"Could not import RAG components: {e}")
        return None, None


@tool()
async def search_evidence_tool(
    query: str, top_k: int = 5, use_web_search: bool = True
) -> str:
    """Search for evidence passages relevant to a query using hybrid retrieval."""
    logger.info(
        "[DEBUG] search_evidence_tool called: query='%s...', top_k=%d",
        query[:50],
        top_k,
    )

    try:
        vector_store_fn, _ = _get_rag_components()
        from trust_agents.rag.web_search import search_web

        results = []
        if vector_store_fn is not None:
            try:
                vector_store = vector_store_fn()
                results = retrieve_with_fallback(
                    query=query,
                    local_search=lambda value: vector_store.similarity_search(
                        value, k=top_k
                    ),
                    web_search=lambda value: search_web(value, num_results=top_k),
                    threshold=settings.similarity_threshold,
                    use_web_search=use_web_search,
                    max_results=top_k,
                )
            except Exception as e:
                logger.warning(f"Vector search failed, falling back to web only: {e}")
                results = search_web(query, num_results=top_k)
        else:
            logger.warning("RAG not available, using web search only")
            results = search_web(query, num_results=top_k)

        if not results:
            logger.warning("No search results found, using mock evidence as last resort")
            evidence = [
                {
                    "id": f"mock_{index + 1:03d}",
                    "text": f"Information regarding '{query[:50]}...'. No real-time evidence found.",
                    "source": "System Fallback",
                    "score": 0.5,
                }
                for index in range(1)
            ]
        else:
            evidence = [
                {
                    "id": result.get("id", f"passage_{index + 1:03d}"),
                    "text": result.get("content", result.get("text", "")),
                    "source": result.get("source", "Web Search"),
                    "score": result.get("score", 0.5),
                    "url": result.get("url", ""),
                    "metadata": result.get("metadata", {}),
                }
                for index, result in enumerate(results)
            ]

        result = {
            "evidence": evidence,
            "query": query,
            "top_k": top_k,
            "total_found": len(evidence),
            "avg_score": (
                sum(item.get("score", 0) for item in evidence) / len(evidence)
                if evidence
                else 0
            ),
        }

        logger.info("search_evidence_tool completed: %d results", len(evidence))
        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Error in search_evidence_tool: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return json.dumps(
            {
                "evidence": [],
                "error": str(e),
                "query": query,
                "top_k": top_k,
            },
            ensure_ascii=False,
        )


def _merge_faiss_web(
    faiss_results: list[dict[str, Any]],
    web_results: list[dict[str, Any]],
    max_results: int = 8,
) -> list[dict[str, Any]]:
    """Compatibility wrapper for shared merge logic."""
    return merge_results(faiss_results, web_results, max_results=max_results)


@tool()
async def get_passage_tool(passage_id: str) -> str:
    """Get details of a specific passage by ID."""
    logger.info(f"[DEBUG] get_passage_tool called: passage_id={passage_id}")

    try:
        vector_store_fn, _ = _get_rag_components()

        if vector_store_fn is not None:
            vector_store = vector_store_fn()
            passage = None
            for index, doc in enumerate(vector_store.documents):
                if (
                    doc.get("id") == passage_id
                    or index == int(passage_id.split("_")[-1]) - 1
                ):
                    passage = doc.copy()
                    passage["id"] = passage_id
                    break

            if passage is None:
                passage = {
                    "id": passage_id,
                    "text": f"Passage {passage_id} not found in index.",
                    "source": "Unknown",
                    "metadata": {},
                }
        else:
            passage = {
                "id": passage_id,
                "text": f"Text for passage {passage_id}. Load actual passage from FAISS index.",
                "source": "Unknown",
                "metadata": {},
            }

        logger.info("get_passage_tool completed")
        return json.dumps(passage, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Error in get_passage_tool: {e}")
        return json.dumps({"id": passage_id, "error": str(e)}, ensure_ascii=False)


@tool()
async def list_indexed_documents_tool() -> str:
    """List all indexed documents in the evidence corpus."""
    logger.info("[DEBUG] list_indexed_documents_tool called")

    try:
        vector_store_fn, _ = _get_rag_components()

        if vector_store_fn is not None:
            vector_store = vector_store_fn()
            docs = []
            seen_ids = set()
            for index, doc in enumerate(vector_store.documents):
                doc_id = doc.get("id", f"doc_{index:03d}")
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    docs.append(
                        {
                            "id": doc_id,
                            "title": doc.get("title", doc.get("content", "")[:100]),
                            "source": doc.get("source", "ViFactCheck"),
                            "url": doc.get("url", ""),
                        }
                    )
        else:
            docs = [
                {
                    "id": "doc_001",
                    "title": "ViFactCheck Dataset",
                    "source": "Vietnamese Fact-Check",
                    "passage_count": 5000,
                }
            ]

        result = {"documents": docs, "total_documents": len(docs)}

        logger.info("list_indexed_documents_tool completed: %d documents", len(docs))
        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Error in list_indexed_documents_tool: {e}")
        return json.dumps({"documents": [], "error": str(e)}, ensure_ascii=False)


@tool()
async def index_documents_tool(
    corpus_path: str, chunk_size: int = 160, force_rebuild: bool = False
) -> str:
    """Index documents for evidence retrieval."""
    logger.info(
        "[DEBUG] index_documents_tool called: corpus_path=%s, chunk_size=%d",
        corpus_path,
        chunk_size,
    )

    try:
        from trust_agents.agents.retrieval_agent_core import get_retrieval_system

        retrieval_system = get_retrieval_system(index_dir="retrieval_index")
        stats = retrieval_system.ingest_corpus(
            corpus_path=corpus_path,
            chunk_size=chunk_size,
            force_rebuild=force_rebuild,
        )

        result = {
            "success": True,
            "corpus_path": corpus_path,
            "chunk_size": chunk_size,
            "force_rebuild": force_rebuild,
            **stats,
        }

        logger.info("index_documents_tool completed successfully")
        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Error in index_documents_tool: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return json.dumps(
            {
                "success": False,
                "error": str(e),
                "corpus_path": corpus_path,
            },
            ensure_ascii=False,
        )

"""
Evidence Retrieval Agent - ReAct Agent with hybrid search tools.

Uses BM25 + Dense retrieval (FAISS) to find evidence passages for fact-checking.
Supports OpenAI, Google Gemini (AI Studio), and NVIDIA NIM backends.
"""

import logging
from typing import Any

from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent

from trust_agents.agents.retrieval_agent_tools import (
    get_passage_tool,
    index_documents_tool,
    list_indexed_documents_tool,
    search_evidence_tool,
)
from trust_agents.llm.factory import create_chat_model
from trust_agents.parsing import (
    extract_last_message_text,
    parse_dict_payload,
    parse_evidence_payload,
)

load_dotenv()
logger = logging.getLogger("EvidenceRetriever.Agent")


async def run_evidence_retrieval_agent(
    query: str, top_k: int = 5
) -> list[dict[str, Any]]:
    """Retrieve evidence passages relevant to a query using hybrid search."""
    model = create_chat_model()

    logger.info("[AGENT] Evidence Retrieval Agent initialized")

    agent_prompt = f"""
You are an Evidence Retrieval agent. Your task is to find relevant evidence passages for the given query or claim.

Query: {query}

You have access to the following tools:
- search_evidence_tool: Search for evidence passages using hybrid retrieval (BM25 + semantic search)
- get_passage_tool: Get details of a specific passage by ID
- list_indexed_documents_tool: List all indexed documents

Your GOAL:
Use search_evidence_tool to find the top {top_k} most relevant evidence passages for the query.

After retrieving evidence, return JSON: {{"evidence": [list of passages with text, source, and scores]}}.
""".strip()

    tools = [search_evidence_tool, get_passage_tool, list_indexed_documents_tool]
    logger.info("[AGENT] Loaded %d tools.", len(tools))

    agent = create_react_agent(model, tools)
    logger.info("[AGENT] LangGraph agent initialized. Invoking agent...")

    result: dict[str, Any] = await agent.ainvoke(
        {"messages": [{"role": "user", "content": agent_prompt}]}
    )

    msgs = result.get("messages", [])
    logger.debug("[AGENT] Received %d messages", len(msgs))
    for index, msg in enumerate(msgs):
        content = extract_last_message_text([msg])
        logger.debug(
            "Message %d | %s | len=%d\n%s\n",
            index,
            type(msg).__name__,
            len(content),
            content,
        )

    logger.info("[AGENT] Agent execution completed. Processing results...")
    content = extract_last_message_text(msgs)
    evidence = parse_evidence_payload(content)
    if evidence:
        logger.info(
            "[AGENT] Successfully extracted %d evidence passages", len(evidence)
        )
        return evidence

    parsed = parse_dict_payload(content)
    if parsed and isinstance(parsed.get("evidence"), list):
        evidence = [item for item in parsed["evidence"] if isinstance(item, dict)]
        if evidence:
            logger.info(
                "[AGENT] Successfully extracted %d evidence passages", len(evidence)
            )
            return evidence

    for msg in reversed(msgs):
        message_text = extract_last_message_text([msg])
        evidence = parse_evidence_payload(message_text)
        if evidence:
            logger.info(
                "[AGENT] Recovered %d evidence passages from intermediate message",
                len(evidence),
            )
            return evidence

        parsed = parse_dict_payload(message_text)
        if parsed and isinstance(parsed.get("evidence"), list):
            evidence = [item for item in parsed["evidence"] if isinstance(item, dict)]
            if evidence:
                logger.info(
                    "[AGENT] Recovered %d evidence passages from intermediate message",
                    len(evidence),
                )
                return evidence

    logger.warning("[AGENT] No evidence found")
    return []


def run_evidence_retrieval_agent_sync(
    query: str, top_k: int = 5
) -> list[dict[str, Any]]:
    """Synchronous wrapper for run_evidence_retrieval_agent."""
    import asyncio

    return asyncio.run(run_evidence_retrieval_agent(query, top_k))


async def index_documents_async(
    corpus_path: str, chunk_size: int = 160
) -> dict[str, Any]:
    """Index documents for retrieval."""
    model = create_chat_model()

    agent_prompt = f"""
You are an indexing agent. Index the documents at the given path.

Corpus path: {corpus_path}
Chunk size: {chunk_size}

Use index_documents_tool to index the documents and report the results.
""".strip()

    tools = [index_documents_tool]
    agent = create_react_agent(model, tools)

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": agent_prompt}]}
    )

    parsed = parse_dict_payload(extract_last_message_text(result.get("messages", [])))
    if isinstance(parsed, dict):
        return parsed

    return {"success": False, "error": "Failed to parse indexing result"}


def index_documents_sync(corpus_path: str, chunk_size: int = 160) -> dict[str, Any]:
    """Synchronous wrapper for index_documents_async."""
    import asyncio

    return asyncio.run(index_documents_async(corpus_path, chunk_size))

"""
Claim Extractor Agent - ReAct Agent with NLP tools.

Uses Named Entity Recognition, Dependency Parsing, and LLM to extract factual claims.
Supports OpenAI, Google Gemini (AI Studio), and NVIDIA NIM backends.
"""

import json
import logging
from typing import Any

from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent

from trust_agents.agents.claim_extractor_tools import (
    dependency_claim_extraction_tool,
    llm_claim_extraction_tool,
    ner_claim_extraction_tool,
)
from trust_agents.llm.factory import create_chat_model
from trust_agents.parsing import extract_last_message_text, parse_claims_payload

load_dotenv()
logger = logging.getLogger("ClaimExtractor.Agent")


async def run_claim_extractor_agent(text: str) -> list[str]:
    """Extract claims from text using NER, dependency parsing, and LLM tools."""
    model = create_chat_model()

    state = {
        "text": text,
        "ner_done": False,
        "dependency_done": False,
        "llm_done": False,
    }
    logger.info("[AGENT] Initial state initialized")
    logger.info("[DEBUG] State : %s", state)

    agent_prompt = f"""
You are a Claim Extractor agent. Identify factual claims or key assertions within text using the tools provided.

Text: {text}

State: {json.dumps(state, indent=2)}

You have access to three tools:
- ner_claim_extraction_tool: Extract claims using Named Entity Recognition
- dependency_claim_extraction_tool: Extract claims using dependency parsing
- llm_claim_extraction_tool: Extract claims using LLM zero-shot reasoning

Your GOAL:
Extract all factual claims from the text using all available tools.

When all tools have been used, combine all results, remove duplicates, and return JSON: {{"claims": ["claim1", "claim2", ...]}}.
""".strip()  # noqa: E501

    tools = [
        ner_claim_extraction_tool,
        dependency_claim_extraction_tool,
        llm_claim_extraction_tool,
    ]
    logger.info("[AGENT] Loaded %d tools.", len(tools))

    agent = create_react_agent(model, tools)
    logger.info("[AGENT] LangGraph agent initialized. Invoking agent...")

    result: dict[str, Any] = await agent.ainvoke(
        {"messages": [{"role": "user", "content": agent_prompt}], **state}
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
    claims = parse_claims_payload(extract_last_message_text(msgs))
    if claims:
        logger.info("[AGENT] Successfully extracted %d claims", len(claims))
        return claims

    recovered_claims: list[str] = []
    seen_claims: set[str] = set()
    for msg in reversed(msgs):
        message_claims = parse_claims_payload(extract_last_message_text([msg]))
        for claim in message_claims:
            normalized_claim = claim.strip()
            if normalized_claim and normalized_claim not in seen_claims:
                seen_claims.add(normalized_claim)
                recovered_claims.append(normalized_claim)

    if recovered_claims:
        logger.info(
            "[AGENT] Recovered %d claims from intermediate messages",
            len(recovered_claims),
        )
        return recovered_claims

    logger.warning("[AGENT] No claims found")
    return []


def run_claim_extractor_agent_sync(text: str) -> list[str]:
    """Synchronous wrapper for run_claim_extractor_agent."""
    import asyncio

    return asyncio.run(run_claim_extractor_agent(text))

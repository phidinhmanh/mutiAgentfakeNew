
"""
Explainer Agent - ReAct Agent for generating explanations.

Creates natural language explanations of verification results with evidence citations.
Supports OpenAI, Google Gemini (AI Studio), and NVIDIA NIM backends.
"""

import json
import logging
from typing import Any

from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent

from trust_agents.agents.explainer_tools import (
    cite_evidence_tool,
    create_report_tool,
    generate_explanation_tool,
    summarize_verification_tool,
)
from trust_agents.llm.factory import create_chat_model
from trust_agents.parsing import extract_last_message_text, parse_dict_payload

load_dotenv()
logger = logging.getLogger("Explainer.Agent")


async def run_explainer_agent(
    claim: str,
    verdict_data: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    """Generate explanation for verification result."""
    model = create_chat_model()

    logger.info("[AGENT] Explainer Agent initialized")

    verdict_json = json.dumps(verdict_data, ensure_ascii=False)
    evidence_json = json.dumps(evidence[:5], ensure_ascii=False)

    verdict = verdict_data.get("verdict", "uncertain")
    confidence = verdict_data.get("confidence", 0.5)

    agent_prompt = f"""You are an Explainer agent. Your task is to create a clear, comprehensive explanation of the fact-check result.

Claim: {claim}
Verdict: {verdict} (confidence: {confidence:.1%})

Verdict Data: {verdict_json[:500]}...
Evidence Data: {evidence_json[:500]}...

You have access to these tools:
- summarize_verification_tool: Create a concise summary
- generate_explanation_tool: Generate detailed explanation with citations
- cite_evidence_tool: Format evidence citations
- create_report_tool: Compile comprehensive report

Your PROCESS:
1. Use summarize_verification_tool to create a summary
2. Use cite_evidence_tool to format citations
3. Use generate_explanation_tool to create detailed explanation
4. Use create_report_tool to compile everything

After creating the report, return JSON with the complete fact-check report.
""".strip()

    tools = [
        summarize_verification_tool,
        generate_explanation_tool,
        cite_evidence_tool,
        create_report_tool,
    ]
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
    parsed = parse_dict_payload(extract_last_message_text(msgs))
    if parsed:
        logger.info("[AGENT] Successfully extracted report")
        return parsed

    logger.warning("[AGENT] No report found, creating basic report")
    return {
        "claim": claim,
        "verdict": verdict,
        "confidence": confidence,
        "label": verdict,
        "summary": f"The claim is {verdict} with {confidence:.1%} confidence.",
        "explanation": verdict_data.get("reasoning", "Based on available evidence."),
        "evidence_count": len(evidence),
    }


def run_explainer_agent_sync(
    claim: str,
    verdict_data: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    """Synchronous wrapper for run_explainer_agent."""
    import asyncio

    return asyncio.run(run_explainer_agent(claim, verdict_data, evidence))

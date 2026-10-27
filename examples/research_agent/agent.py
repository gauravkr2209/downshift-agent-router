"""Multi-step Research Agent using Downshift recorder.

Tags used: plan, search_query, summarize_page, final_answer.
Enforces:
- 40-step cap
- max_tokens: 3000 for plan & final_answer, 1500 for other steps
- temperature=0.2 explicitly set on every call
- short numbered plan only in plan prompt
- 2-3 searches before summarizing (~10+ steps per run)
- 2000-character tool result truncation with logging
- keyword group checklist evaluation and Run.meta['checklist_score'] storage
"""

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from sqlmodel import Session

from examples.research_agent.checklist import (
    CHECKLIST_ITEMS,
    RESEARCH_QUESTION,
    ChecklistGroup,
    evaluate_checklist_compliance,
)
from sdk import downshift
from server.app import db as db_mod
from server.app.models import Run
from server.app.router import load_routing_config

logger = logging.getLogger("downshift.research_agent")

MAX_AGENT_STEPS = 40
PLAN_MAX_TOKENS = 3000
FINAL_ANSWER_MAX_TOKENS = 3000
DEFAULT_MAX_TOKENS = 1500
DEFAULT_TEMPERATURE = 0.2
TOOL_RESULT_MAX_CHARS = 2000
DEFAULT_MAX_SEARCHES = 3

# One plan + one final answer + (one query and two summaries per search).
MAX_SEARCHES_BY_STEP_CAP = (MAX_AGENT_STEPS - 2) // 3


def truncate_tool_result(content: str) -> str:
    """Truncate tool result to 2000 characters and log when truncation occurs."""
    if len(content) > TOOL_RESULT_MAX_CHARS:
        orig_len = len(content)
        truncated = content[:TOOL_RESULT_MAX_CHARS]
        logger.info(
            "Truncated tool result from %d to %d characters.",
            orig_len,
            TOOL_RESULT_MAX_CHARS,
        )
        return truncated
    return content


def sanitize_history_message(msg: Any) -> dict[str, Any]:
    """Extract only role, content, and tool_calls for sending back to the model."""
    if hasattr(msg, "model_dump"):
        raw = msg.model_dump()
    elif isinstance(msg, dict):
        raw = msg
    else:
        raw = {
            "role": getattr(msg, "role", "assistant"),
            "content": getattr(msg, "content", ""),
            "tool_calls": getattr(msg, "tool_calls", None),
        }

    clean: dict[str, Any] = {
        "role": raw.get("role", "assistant"),
        "content": raw.get("content") or "",
    }
    if raw.get("tool_calls"):
        clean["tool_calls"] = raw["tool_calls"]
    return clean


class _MockUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class _MockMessage(BaseModel):
    role: str = "assistant"
    content: str
    reasoning_content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class _MockChoice(BaseModel):
    message: _MockMessage
    index: int = 0
    finish_reason: str = "stop"


class _MockResponse(BaseModel):
    id: str = "mock-cmpl"
    model: str = "nemotron_ultra"
    usage: _MockUsage
    choices: list[_MockChoice]


class MockSearchAgentClient:
    """OpenAI-compatible mock client that simulates the research workflow."""

    def __init__(self, model: str = "nemotron_ultra") -> None:
        self.model = model
        self.chat = self
        self.completions = self
        self.last_received_messages: list[dict[str, Any]] = []

    def create(self, *args: Any, **kwargs: Any) -> _MockResponse:
        """Return a deterministic simulated response for the latest user prompt."""
        self.last_received_messages = kwargs.get("messages", [])
        messages = self.last_received_messages
        last_user_content = ""
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "user":
                last_user_content = str(message.get("content", "")).lower()
                break

        # Check specific prompt types by intent:
        # 1. Final synthesis prompt
        # 2. Document summarization prompt (which may quote previous query text)
        # 3. Search query generation prompt (which mentions the plan)
        # 4. Initial plan prompt
        if (
            last_user_content.startswith("synthesize")
            or "definitive final answer" in last_user_content
        ):
            # Extract distinct summary findings from prior assistant messages in conversation
            summaries = []
            for m in messages:
                if isinstance(m, dict) and m.get("role") == "assistant":
                    c = str(m.get("content", ""))
                    if c.startswith("Summary of page:") and c not in summaries:
                        summaries.append(c)

            if summaries:
                bullet_points = "\n".join(
                    f"- {s.removeprefix('Summary of page:').strip()}" for s in summaries
                )
                content = (
                    "Synthesized final answer covering essential facts and technical specifications "
                    "based on simulated mock research:\n\n"
                    "[SIMULATED MOCK RESEARCH - UNVERIFIED DATA]\n"
                    f"Key findings from simulated research steps:\n{bullet_points}\n\n"
                    "Synthesis Summary: Nebius Token Factory hosts NVIDIA Nemotron models structured across "
                    "Ultra, Super, and Nano parameter tiers (from 4B to 340B). These models provide dedicated "
                    "capabilities for autonomous AI agent workloads, including function calling, tool use, "
                    "and multi-step reasoning.\n\n"
                    "Note: All findings above represent simulated mock outputs for workflow validation and "
                    "have not been independently verified against live endpoints."
                )
                reasoning = "Synthesized all accumulated page summaries into comprehensive report."
            else:
                content = (
                    "Synthesized final answer covering essential facts and technical specifications.\n\n"
                    "[SIMULATED MOCK RESEARCH - UNVERIFIED DATA]\n"
                    "Nebius Token Factory hosts NVIDIA Nemotron models in Ultra, Super, and Nano tiers "
                    "(4B to 340B) optimized for AI agent workloads with function calling, tool use, and reasoning.\n"
                    "Note: Simulated output for testing purposes only."
                )
                reasoning = "Compiled standard simulated synthesis covering essential facts and specifications."

        elif (
            last_user_content.startswith("summarize") or "summarize key facts" in last_user_content
        ):
            if "nebius" in last_user_content or "token factory" in last_user_content:
                content = (
                    "Summary of page: Nebius Token Factory provides hosted access to NVIDIA Nemotron "
                    "models via OpenAI-compatible endpoints."
                )
                reasoning = (
                    "Extracted model availability and platform hosting details from document."
                )
            elif (
                "tier" in last_user_content
                or "architecture" in last_user_content
                or "ultra" in last_user_content
                or "super" in last_user_content
                or "nano" in last_user_content
            ):
                content = (
                    "Summary of page: Nemotron architecture features Ultra, Super, and Nano tiers with "
                    "technical specifications from 4B to 340B."
                )
                reasoning = "Extracted architecture tiers and technical parameter specifications from document."
            elif (
                "agent" in last_user_content
                or "tool" in last_user_content
                or "function" in last_user_content
                or "reasoning" in last_user_content
            ):
                content = (
                    "Summary of page: Nemotron models support enterprise AI agent workflows with "
                    "function calling, tool use, and reasoning."
                )
                reasoning = (
                    "Extracted agent capabilities, tool use, and reasoning features from document."
                )
            else:
                content = (
                    "Summary of page: contains verified technical specifications and findings covering "
                    "essential facts."
                )
                reasoning = "Distilling main points from search document."

        elif "search query" in last_user_content:
            query_idx = None
            for num in (1, 2, 3, 4, 5):
                if f"#{num}" in last_user_content or f"query {num}" in last_user_content:
                    query_idx = num - 1
                    break
            if query_idx is None:
                query_idx = sum(
                    1
                    for m in messages
                    if isinstance(m, dict)
                    and m.get("role") == "assistant"
                    and str(m.get("content", "")).startswith("Search query:")
                )

            queries = [
                "Search query: NVIDIA Nemotron models hosted on Nebius Token Factory availability",
                "Search query: Nemotron architecture tiers specifications ultra super nano parameter scale",
                "Search query: Nemotron AI agent capabilities function calling reasoning tool use",
            ]
            if query_idx < len(queries):
                content = queries[query_idx]
            else:
                content = (
                    f"Search query: Nemotron technical specifications iteration {query_idx + 1}"
                )
            reasoning = f"Refining search query #{query_idx + 1} based on research plan."

        elif "numbered plan only" in last_user_content or "plan" in last_user_content:
            content = (
                "1. Identify NVIDIA Nemotron models and check availability on Nebius Token Factory.\n"
                "2. Examine architecture tiers, parameter specifications, and deployment options (Ultra, Super, Nano).\n"
                "3. Investigate enterprise AI agent workloads including tool use, function calling, and reasoning."
            )
            reasoning = "Drafting concise numbered research plan without conversational filler."

        else:
            content = (
                "Synthesized final answer covering essential facts and technical specifications."
            )
            reasoning = "Compiling all summaries into final comprehensive report."

        return _MockResponse(
            model=kwargs.get("model", self.model),
            usage=_MockUsage(
                prompt_tokens=150,
                completion_tokens=60,
                total_tokens=210,
            ),
            choices=[
                _MockChoice(
                    message=_MockMessage(
                        content=content,
                        reasoning_content=reasoning,
                    ),
                    finish_reason="stop",
                )
            ],
        )


class ResearchAgent:
    """Multi-step research agent with tagged steps and strict limits."""

    def __init__(
        self,
        client: Any,
        baseline_model: str = "nemotron_ultra",
        cost_ceiling_usd: float | None = 0.20,
        temperature: float = DEFAULT_TEMPERATURE,
        search_fn: Callable[[str], list[str]] | None = None,
        search_engine_name: str = "Mock",
        routing_config: str | Path | dict[str, str] | None = None,
    ) -> None:
        self.raw_client = client
        self.client = downshift.wrap_client(client)
        self.baseline_model = baseline_model
        self.cost_ceiling_usd = cost_ceiling_usd
        self.temperature = temperature
        self.search_fn = search_fn or self._default_search
        self.search_engine_name = search_engine_name
        self.routing_config = routing_config
        self.routing_map = load_routing_config(routing_config)

    def get_model_for_tag(self, tag: str) -> str:
        """Select configured model for a tag, falling back to baseline_model."""
        if not tag or tag not in self.routing_map:
            return self.baseline_model
        return self.routing_map[tag]

    def _default_search(self, query: str) -> list[str]:
        """Placeholder search returning distinct mock source snippets for testing."""
        q_lower = query.lower()
        if "nebius" in q_lower or "token factory" in q_lower or "availability" in q_lower:
            return [
                (
                    f"[SIMULATED MOCK SOURCE - UNVERIFIED] Source 1 for '{query}': "
                    "Nebius Token Factory documentation lists hosted NVIDIA Nemotron models "
                    "available via OpenAI-compatible API endpoints."
                ),
                (
                    f"[SIMULATED MOCK SOURCE - UNVERIFIED] Source 2 for '{query}': "
                    "Catalog records confirm access to Nemotron deployment options on Nebius cloud infrastructure."
                ),
            ]
        elif (
            "tier" in q_lower
            or "architecture" in q_lower
            or "specification" in q_lower
            or "ultra" in q_lower
            or "super" in q_lower
            or "nano" in q_lower
        ):
            return [
                (
                    f"[SIMULATED MOCK SOURCE - UNVERIFIED] Source 1 for '{query}': "
                    "Architecture specifications detail Nemotron Ultra, Super, and Nano parameter tiers "
                    "from 4B/30B to 120B/340B."
                ),
                (
                    f"[SIMULATED MOCK SOURCE - UNVERIFIED] Source 2 for '{query}': "
                    "High-throughput FP8 specifications and context windows are optimized for large-scale inference."
                ),
            ]
        elif (
            "agent" in q_lower
            or "tool" in q_lower
            or "function" in q_lower
            or "reasoning" in q_lower
        ):
            return [
                (
                    f"[SIMULATED MOCK SOURCE - UNVERIFIED] Source 1 for '{query}': "
                    "Agent capabilities include multi-turn function calling, structured tool use, and multi-step reasoning."
                ),
                (
                    f"[SIMULATED MOCK SOURCE - UNVERIFIED] Source 2 for '{query}': "
                    "Models support synthetic data generation, complex tool pipelines, and agentic RAG workflows."
                ),
            ]
        else:
            return [
                f"[SIMULATED MOCK SOURCE - UNVERIFIED] Source 1 for '{query}': contains essential facts and core findings.",
                f"[SIMULATED MOCK SOURCE - UNVERIFIED] Source 2 for '{query}': contains technical specifications and details.",
            ]

    def run(
        self,
        question: str = RESEARCH_QUESTION,
        checklist: Sequence[ChecklistGroup] | None = None,
        max_searches: int = DEFAULT_MAX_SEARCHES,
    ) -> dict[str, Any]:
        """Execute the multi-step research workflow within a Downshift recorded run."""
        if max_searches < 0:
            raise ValueError("max_searches must be zero or greater")

        search_count = min(max_searches, MAX_SEARCHES_BY_STEP_CAP)
        if search_count != max_searches:
            logger.warning(
                "Capped max_searches from %d to %d to respect the %d-step limit.",
                max_searches,
                search_count,
                MAX_AGENT_STEPS,
            )

        task_name = "research_task"
        agent_name = "research_agent_a"
        target_checklist = checklist if checklist is not None else CHECKLIST_ITEMS

        search_stats: list[dict[str, Any]] = []

        with downshift.run(
            task_name=task_name,
            agent_name=agent_name,
            baseline_model=self.baseline_model,
            cost_ceiling_usd=self.cost_ceiling_usd,
        ) as active_run:
            conversation_history: list[dict[str, Any]] = [
                {
                    "role": "system",
                    "content": (
                        "You are an autonomous research agent. Be thorough, accurate, and concise. "
                        "Distinguish verified evidence from uncertainty."
                    ),
                }
            ]

            # 1. Step: Plan (max_tokens=3000)
            conversation_history.append(
                {
                    "role": "user",
                    "content": (
                        f"Formulate a short numbered plan only for researching: {question}. "
                        "Do not include any conversational preamble or outro."
                    ),
                }
            )
            plan_resp = self.client.chat.completions.create(
                model=self.get_model_for_tag("plan"),
                messages=conversation_history,
                max_tokens=PLAN_MAX_TOKENS,
                temperature=self.temperature,
                downshift_tag="plan",
            )
            plan_msg = plan_resp.choices[0].message
            conversation_history.append(sanitize_history_message(plan_msg))

            # 2. Step: 2-3 searches with page summaries (~10+ steps total)
            for i in range(search_count):
                conversation_history.append(
                    {
                        "role": "user",
                        "content": f"Formulate search query #{i + 1} based on your plan.",
                    }
                )
                query_resp = self.client.chat.completions.create(
                    model=self.get_model_for_tag("search_query"),
                    messages=conversation_history,
                    max_tokens=DEFAULT_MAX_TOKENS,
                    temperature=self.temperature,
                    downshift_tag="search_query",
                )
                query_msg = query_resp.choices[0].message
                conversation_history.append(sanitize_history_message(query_msg))

                # Execute search and record stats
                raw_sources = self.search_fn(query_msg.content)
                search_stats.append(
                    {
                        "search_index": i + 1,
                        "query": query_msg.content,
                        "engine": self.search_engine_name,
                        "results_count": len(raw_sources),
                    }
                )
                print(
                    f"[{self.search_engine_name}] Search #{i + 1} for '{query_msg.content[:60]}' "
                    f"returned {len(raw_sources)} results."
                )

                for raw_src in raw_sources:
                    # Truncate every tool result to 2000 chars before adding to history
                    src = truncate_tool_result(raw_src)
                    conversation_history.append(
                        {
                            "role": "user",
                            "content": f"Summarize key facts from document: {src}",
                        }
                    )
                    sum_resp = self.client.chat.completions.create(
                        model=self.get_model_for_tag("summarize_page"),
                        messages=conversation_history,
                        max_tokens=DEFAULT_MAX_TOKENS,
                        temperature=self.temperature,
                        downshift_tag="summarize_page",
                    )
                    sum_msg = sum_resp.choices[0].message
                    conversation_history.append(sanitize_history_message(sum_msg))

            # 3. Step: Final Answer (max_tokens=3000)
            conversation_history.append(
                {
                    "role": "user",
                    "content": (
                        f"Synthesize all research above into a definitive final answer for: {question}"
                    ),
                }
            )
            final_resp = self.client.chat.completions.create(
                model=self.get_model_for_tag("final_answer"),
                messages=conversation_history,
                max_tokens=FINAL_ANSWER_MAX_TOKENS,
                temperature=self.temperature,
                downshift_tag="final_answer",
            )
            final_msg = final_resp.choices[0].message
            final_answer = final_msg.content

            # 4. Objective success check against checklist groups
            eval_result = evaluate_checklist_compliance(final_answer, target_checklist)
            task_success = eval_result["all_passed"]
            checklist_score = eval_result["checklist_score"]

            # Save task_success and checklist_score in Run.meta
            with Session(db_mod.engine) as session:
                db_run = session.get(Run, active_run.id)
                if db_run:
                    db_run.task_success = task_success
                    meta = dict(db_run.meta) if db_run.meta else {}
                    meta["checklist_score"] = checklist_score
                    db_run.meta = meta
                    session.add(db_run)
                    session.commit()

        return {
            "run_id": active_run.id,
            "final_answer": final_answer,
            "evaluation": eval_result,
            "task_success": task_success,
            "checklist_score": checklist_score,
            "search_stats": search_stats,
            "search_engine_name": self.search_engine_name,
            "conversation_history": conversation_history,
            "routing_map": self.routing_map,
        }

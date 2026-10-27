"""Baseline execution runner for Research Agent A.

Defaults to 1 run against mock client. Real API execution is behind
an explicit --real flag.
Enforces max_tokens: 3000 for plan & final_answer, 1500 for other steps.
Enforces 40-step cap, and default cost ceiling of $0.20.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

# Ensure repository root is in sys.path when executed directly as a script
_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from openai import OpenAI
from sqlmodel import Session, col, select

from examples.research_agent.agent import (
    DEFAULT_MAX_SEARCHES,
    DEFAULT_MAX_TOKENS,
    PLAN_MAX_TOKENS,
    MockSearchAgentClient,
    ResearchAgent,
)
from examples.research_agent.checklist import (
    CHECKLIST_ITEMS,
    RESEARCH_QUESTION,
)
from server.app import db as db_mod
from server.app.cli import inspect_run
from server.app.config import settings
from server.app.db import create_db_and_tables, resolve_model_price, sync_prices_from_yaml
from server.app.models import Step


def create_tavily_search_fn(api_key: str | None) -> Any:
    """Create Tavily web search function if API key is configured."""
    if not api_key or "PLACEHOLDER" in api_key.upper() or api_key.startswith("your_"):
        return None

    try:
        from tavily import TavilyClient

        tavily_client = TavilyClient(api_key=api_key)

        def search_fn(query: str) -> list[str]:
            res = tavily_client.search(query=query, max_results=2)
            return [str(r.get("content", "")) for r in res.get("results", [])]

        return search_fn
    except ImportError:
        return None


def run_baseline(
    runs: int = 1,
    use_real_api: bool = False,
    baseline_model: str = "nemotron_ultra",
    cost_ceiling_usd: float = 0.20,
    routing_config: str | None = None,
) -> None:
    """Execute baseline runs with Downshift recording."""
    create_db_and_tables()
    sync_prices_from_yaml()

    if use_real_api:
        # Validate that research question and checklist items are populated
        if not RESEARCH_QUESTION or "PLACEHOLDER" in RESEARCH_QUESTION.upper():
            print(
                "REFUSAL: RESEARCH_QUESTION in checklist.py is still a placeholder.",
                file=sys.stderr,
            )
            sys.exit(1)

        if not CHECKLIST_ITEMS:
            print(
                "REFUSAL: CHECKLIST_ITEMS in checklist.py is empty.",
                file=sys.stderr,
            )
            sys.exit(1)

        # Validate real environment credentials
        if not settings.nebius_api_key or "PLACEHOLDER" in settings.nebius_api_key.upper():
            print("REFUSAL: NEBIUS_API_KEY is not configured in .env.", file=sys.stderr)
            sys.exit(1)
        if not settings.nebius_base_url or "PLACEHOLDER" in settings.nebius_base_url.upper():
            print("REFUSAL: NEBIUS_BASE_URL is not configured in .env.", file=sys.stderr)
            sys.exit(1)

        with Session(db_mod.engine) as session:
            try:
                target_model_id = settings.ultra_model_id or baseline_model
                real_model, price = resolve_model_price(target_model_id, session)
            except (ValueError, KeyError, FileNotFoundError) as e:
                print(f"REFUSAL: Pricing resolution failed: {e}", file=sys.stderr)
                sys.exit(1)

            if price.input_per_mtok is None or price.output_per_mtok is None:
                print(f"REFUSAL: Pricing for '{real_model}' has null values.", file=sys.stderr)
                sys.exit(1)

        is_verified, reason = db_mod.check_live_pricing_verified()
        if not is_verified:
            print(f"REFUSAL: {reason}", file=sys.stderr)
            sys.exit(1)

        search_fn = create_tavily_search_fn(settings.tavily_api_key)
        search_engine_name = "Tavily" if search_fn is not None else "MockFallback"
        print(f"Connecting to real API: {settings.nebius_base_url} [Model: {real_model}]")
        print(f"Search Provider: {search_engine_name}")

        client: Any = OpenAI(
            api_key=settings.nebius_api_key,
            base_url=settings.nebius_base_url,
            timeout=settings.nebius_request_timeout,
            max_retries=settings.nebius_max_retries,
        )
    else:
        # Mock mode (default)
        real_model = baseline_model
        search_engine_name = "Mock"
        print(
            f"Running in MOCK mode [Model: {real_model}, max_tokens={DEFAULT_MAX_TOKENS} (Plan/Final={PLAN_MAX_TOKENS})]"
        )
        print(f"Search Provider: {search_engine_name}")
        client = MockSearchAgentClient(model=real_model)
        search_fn = None

    agent = ResearchAgent(
        client=client,
        baseline_model=real_model,
        cost_ceiling_usd=cost_ceiling_usd,
        search_fn=search_fn,
        search_engine_name=search_engine_name,
        routing_config=routing_config,
    )

    for i in range(runs):
        print(f"\n==================== Baseline Run #{i + 1} of {runs} ====================")
        # 1. Print research question and number of checklist items loaded
        print(f"Research Question        : {RESEARCH_QUESTION}")
        print(f"Checklist Items Loaded   : {len(CHECKLIST_ITEMS)}")
        print(f"Cost Ceiling             : ${cost_ceiling_usd:.4f}")

        result = agent.run(
            question=RESEARCH_QUESTION,
            checklist=CHECKLIST_ITEMS,
            max_searches=DEFAULT_MAX_SEARCHES,
        )
        run_id = result["run_id"]
        eval_info = result["evaluation"]
        task_success = result["task_success"]
        final_answer = result["final_answer"]
        search_stats = result.get("search_stats", [])

        # 2. Per step, print finish_reason. Flag any step where finish_reason == "length"
        with Session(db_mod.engine) as session:
            steps = session.exec(
                select(Step).where(Step.run_id == run_id).order_by(col(Step.idx))
            ).all()

            print("\nPer-Step Execution Summary:")
            for s in steps:
                resp = s.response or {}
                choices = resp.get("choices", [])
                finish_reason = choices[0].get("finish_reason", "unknown") if choices else "unknown"
                length_flag = (
                    " [FLAG: finish_reason='length' - output truncated!]"
                    if finish_reason == "length"
                    else ""
                )

                print(
                    f"  Step {s.idx:02d} [{s.tag:<15}] finish_reason: {finish_reason:<8}{length_flag} | "
                    f"Tokens: {s.tokens_in:>4} in / {s.tokens_out:>4} out | Cost: ${s.cost_usd:.6f}"
                )

        # 3. Print whether search used real Tavily or mock, and results returned
        print(f"\nSearch Execution ({result['search_engine_name']}):")
        if search_stats:
            for st in search_stats:
                print(
                    f"  Search #{st['search_index']} ('{st['query'][:50]}'): returned {st['results_count']} result(s)"
                )
        else:
            print("  No searches executed.")

        # 4. Print final answer (first 3000 chars)
        print("\n--- Final Answer (first 3000 characters) ---")
        truncated_answer = final_answer[:3000]
        print(truncated_answer)
        if len(final_answer) > 3000:
            print(f"... [Truncated. Total length: {len(final_answer)} characters]")

        # 5. Print checklist result per item
        print("\n--- Checklist Results per Item ---")
        for group_repr, d in eval_info.get("details", {}).items():
            status = "[FOUND]" if d["matched"] else "[MISSING]"
            matched_info = f"(matched keyword: '{d['keyword']}')" if d["matched"] else ""
            print(f"  {status} {group_repr} {matched_info}")

        print(f"\nChecklist Score : {eval_info['checklist_score']:.2%}")
        print(f"Task Success    : {task_success}")

        print("\nCLI Integrity Verification:")
        verified = inspect_run(run_id)
        if not verified:
            print(f"Integrity check failed for run {run_id}!", file=sys.stderr)
            sys.exit(1)


def main() -> None:
    """CLI entrypoint for running baseline."""
    parser = argparse.ArgumentParser(description="Run baseline research agent")
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of baseline runs to execute (default: 1)",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Connect to real Nebius Token Factory API (default: False/Mock)",
    )
    parser.add_argument(
        "--model",
        default="nemotron_ultra",
        help="Baseline model ID or alias (default: nemotron_ultra)",
    )
    parser.add_argument(
        "--cost-ceiling",
        type=float,
        default=0.20,
        help="Per-run cost ceiling in USD (default: 0.20)",
    )
    parser.add_argument(
        "--routing-config",
        default=None,
        help="Optional path to routing configuration JSON file (default: None)",
    )

    args = parser.parse_args()

    run_baseline(
        runs=args.runs,
        use_real_api=args.real,
        baseline_model=args.model,
        cost_ceiling_usd=args.cost_ceiling,
        routing_config=args.routing_config,
    )


if __name__ == "__main__":
    main()

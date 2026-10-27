"""Side-by-side comparison runner for baseline vs optimized model routing.

Executes the Research Agent under baseline routing (all steps on baseline model)
and optimized routing (per-tag model selection from a routing configuration).
Reports actual recorded costs, token usage, checklist compliance, and per-step
model selection.

DISCLAIMER:
All results produced in default mock mode represent simulated agent interactions.
They verify runtime routing mechanics only and do not establish production model
quality or verified real-world cost savings.
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
    MockSearchAgentClient,
    ResearchAgent,
)
from examples.research_agent.checklist import (
    CHECKLIST_ITEMS,
    RESEARCH_QUESTION,
)
from examples.research_agent.run_baseline import create_tavily_search_fn
from server.app import db as db_mod
from server.app.config import settings
from server.app.db import create_db_and_tables, resolve_model_price, sync_prices_from_yaml
from server.app.models import Run, Step

SIMULATION_DISCLAIMER = (
    "Disclaimer: Results are recorded from simulated mock runs to verify runtime "
    "routing mechanics. Step-level replay estimates do not guarantee compound "
    "multi-turn quality or verified production cost savings."
)


def _get_run_steps(run_id: str, session: Session) -> list[Step]:
    """Retrieve all steps for a run ordered by step index."""
    return list(
        session.exec(select(Step).where(Step.run_id == run_id).order_by(col(Step.idx))).all()
    )


def compare_research_routing(
    routing_config: str | Path | dict[str, str] = "routing-test.json",
    baseline_model: str = "nemotron_ultra",
    cost_ceiling_usd: float = 0.20,
    use_real_api: bool = False,
    print_output: bool = True,
    baseline_run_id: str | None = None,
    optimized_run_id: str | None = None,
) -> dict[str, Any]:
    """Run baseline and optimized agent executions, comparing recorded metrics."""
    create_db_and_tables()
    sync_prices_from_yaml()

    b_run: Run | None = None
    o_run: Run | None = None
    b_steps: list[Step] = []
    o_steps: list[Step] = []
    real_model: str = baseline_model
    routing_map: dict[str, str] = {}
    b_task_success: bool = True
    o_task_success: bool = True
    b_checklist_score: float = 1.0
    o_checklist_score: float = 1.0

    if baseline_run_id and optimized_run_id:
        with Session(db_mod.engine) as session:
            b_run = session.get(Run, baseline_run_id)
            o_run = session.get(Run, optimized_run_id)
            if not b_run or not o_run:
                raise ValueError(
                    f"Specified run IDs not found: baseline='{baseline_run_id}', optimized='{optimized_run_id}'"
                )
            b_steps = _get_run_steps(b_run.id, session)
            o_steps = _get_run_steps(o_run.id, session)
            real_model = b_run.baseline_model
            b_task_success = b_run.task_success if b_run.task_success is not None else True
            o_task_success = o_run.task_success if o_run.task_success is not None else True
            b_checklist_score = float((b_run.meta or {}).get("checklist_score", 1.0))
            o_checklist_score = float((o_run.meta or {}).get("checklist_score", 1.0))

            if routing_config:
                try:
                    from server.app.router import load_routing_config

                    routing_map = load_routing_config(routing_config)
                except (ValueError, KeyError, FileNotFoundError):
                    routing_map = {}
    else:
        # Preflight model and search provider
        if use_real_api:
            if not RESEARCH_QUESTION or "PLACEHOLDER" in RESEARCH_QUESTION.upper():
                raise ValueError("RESEARCH_QUESTION in checklist.py is a placeholder.")
            if not CHECKLIST_ITEMS:
                raise ValueError("CHECKLIST_ITEMS in checklist.py is empty.")
            if not settings.nebius_api_key or "PLACEHOLDER" in settings.nebius_api_key.upper():
                raise ValueError("NEBIUS_API_KEY is not configured in .env.")
            if not settings.nebius_base_url or "PLACEHOLDER" in settings.nebius_base_url.upper():
                raise ValueError("NEBIUS_BASE_URL is not configured in .env.")

            with Session(db_mod.engine) as session:
                target_model_id = settings.ultra_model_id or baseline_model
                real_model, price = resolve_model_price(target_model_id, session)
                if price.input_per_mtok is None or price.output_per_mtok is None:
                    raise ValueError(f"Pricing for '{real_model}' has null values.")

            is_verified, reason = db_mod.check_live_pricing_verified()
            if not is_verified:
                raise ValueError(f"REFUSAL: {reason}")

            search_fn = create_tavily_search_fn(settings.tavily_api_key)
            search_engine_name = "Tavily" if search_fn is not None else "MockFallback"
            client: Any = OpenAI(
                api_key=settings.nebius_api_key,
                base_url=settings.nebius_base_url,
                timeout=settings.nebius_request_timeout,
                max_retries=settings.nebius_max_retries,
            )
        else:
            real_model = baseline_model
            search_engine_name = "Mock"
            client = MockSearchAgentClient(model=real_model)
            search_fn = None

        # 1. Baseline Run (no routing config, all calls use baseline_model)
        baseline_agent = ResearchAgent(
            client=client,
            baseline_model=real_model,
            cost_ceiling_usd=cost_ceiling_usd,
            search_fn=search_fn,
            search_engine_name=search_engine_name,
            routing_config=None,
        )
        baseline_result = baseline_agent.run(
            question=RESEARCH_QUESTION,
            checklist=CHECKLIST_ITEMS,
            max_searches=DEFAULT_MAX_SEARCHES,
        )

        # 2. Optimized Run (per-tag routing configuration applied)
        optimized_agent = ResearchAgent(
            client=client,
            baseline_model=real_model,
            cost_ceiling_usd=cost_ceiling_usd,
            search_fn=search_fn,
            search_engine_name=search_engine_name,
            routing_config=routing_config,
        )
        optimized_result = optimized_agent.run(
            question=RESEARCH_QUESTION,
            checklist=CHECKLIST_ITEMS,
            max_searches=DEFAULT_MAX_SEARCHES,
        )

        b_task_success = baseline_result["task_success"]
        o_task_success = optimized_result["task_success"]
        b_checklist_score = baseline_result["checklist_score"]
        o_checklist_score = optimized_result["checklist_score"]
        routing_map = optimized_agent.routing_map

        # 3. Retrieve DB run records and step details
        with Session(db_mod.engine) as session:
            b_run = session.get(Run, baseline_result["run_id"])
            o_run = session.get(Run, optimized_result["run_id"])

            if not b_run or not o_run:
                raise RuntimeError("Failed to retrieve recorded run records from database.")

            b_steps = _get_run_steps(b_run.id, session)
            o_steps = _get_run_steps(o_run.id, session)

    if not b_run or not o_run:
        raise RuntimeError("Failed to retrieve recorded run records from database.")

    b_cost = b_run.total_cost_usd or 0.0
    o_cost = o_run.total_cost_usd or 0.0
    savings_usd = max(0.0, b_cost - o_cost)
    savings_pct = (savings_usd / b_cost * 100.0) if b_cost > 0.0 else 0.0

    b_tokens = (b_run.tokens_in or 0) + (b_run.tokens_out or 0)
    o_tokens = (o_run.tokens_in or 0) + (o_run.tokens_out or 0)

    # Compile step comparison table
    step_comparisons: list[dict[str, Any]] = []
    max_steps = max(len(b_steps), len(o_steps))
    for idx in range(max_steps):
        bs = b_steps[idx] if idx < len(b_steps) else None
        os = o_steps[idx] if idx < len(o_steps) else None
        step_comparisons.append(
            {
                "idx": idx,
                "tag": (os.tag if os else (bs.tag if bs else "unknown")),
                "baseline_model": bs.model if bs else "N/A",
                "optimized_model": os.model if os else "N/A",
                "baseline_cost_usd": bs.cost_usd if bs else 0.0,
                "optimized_cost_usd": os.cost_usd if os else 0.0,
            }
        )

    comparison_data = {
        "baseline_run_id": b_run.id,
        "optimized_run_id": o_run.id,
        "baseline_model": real_model,
        "routing_map": routing_map,
        "baseline_cost_usd": b_cost,
        "optimized_cost_usd": o_cost,
        "savings_usd": savings_usd,
        "savings_pct": savings_pct,
        "baseline_tokens": b_tokens,
        "optimized_tokens": o_tokens,
        "baseline_steps": len(b_steps),
        "optimized_steps": len(o_steps),
        "baseline_task_success": b_task_success,
        "optimized_task_success": o_task_success,
        "baseline_checklist_score": b_checklist_score,
        "optimized_checklist_score": o_checklist_score,
        "step_comparisons": step_comparisons,
        "disclaimer": SIMULATION_DISCLAIMER,
    }

    if print_output:
        print("\n" + "=" * 80)
        print("          DOWNSHIFT ROUTING COMPARISON: BASELINE vs OPTIMIZED")
        print("=" * 80)
        print(f"MODE: {'REAL API' if use_real_api else 'SIMULATED MOCK RUN'}")
        print(f"NOTICE: {SIMULATION_DISCLAIMER}\n")

        print("--- RUN LEVEL SUMMARY ---")
        print(f"{'Metric':<28} | {'Baseline':<22} | {'Optimized':<22}")
        print("-" * 78)
        print(f"{'Run ID':<28} | {b_run.id:<22} | {o_run.id:<22}")
        print(f"{'Total Cost (USD)':<28} | ${b_cost:<21.6f} | ${o_cost:<21.6f}")
        print(f"{'Cost Reduction':<28} | {'-':<22} | ${savings_usd:.6f} ({savings_pct:.1f}%)")
        print(f"{'Total Tokens':<28} | {b_tokens:<22} | {o_tokens:<22}")
        print(f"{'Total Steps':<28} | {len(b_steps):<22} | {len(o_steps):<22}")
        print(f"{'Checklist Score':<28} | {b_checklist_score:<21.1%} | {o_checklist_score:<21.1%}")
        print(f"{'Task Success':<28} | {b_task_success!s:<22} | {o_task_success!s:<22}")

        print("\n--- PER-STEP MODEL SELECTION & COST ---")
        print(
            f"{'Idx':<4} {'Tag':<16} {'Baseline Model':<32} {'Optimized Model':<32} {'Base Cost':<10} {'Opt Cost':<10}"
        )
        print("-" * 110)
        for sc in step_comparisons:
            bm_short = sc["baseline_model"].split("/")[-1][:30]
            om_short = sc["optimized_model"].split("/")[-1][:30]
            print(
                f"{sc['idx']:<4} {sc['tag']:<16} {bm_short:<32} {om_short:<32} "
                f"${sc['baseline_cost_usd']:<9.5f} ${sc['optimized_cost_usd']:<9.5f}"
            )

        print("\n" + "=" * 80)
        print(f"DISCLAIMER: {SIMULATION_DISCLAIMER}")
        print("=" * 80 + "\n")

    return comparison_data


def main() -> None:
    """CLI entrypoint for routing comparison."""
    parser = argparse.ArgumentParser(
        description="Compare baseline vs optimized model routing in Research Agent"
    )
    parser.add_argument(
        "--routing-config",
        default="routing-test.json",
        help="Path to routing configuration JSON file (default: routing-test.json)",
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
        "--real",
        action="store_true",
        help="Connect to real Nebius Token Factory API (default: False/Mock)",
    )
    parser.add_argument(
        "--baseline-run-id",
        default=None,
        help="Run ID of an existing baseline run to compare (default: None, executes a new baseline run)",
    )
    parser.add_argument(
        "--optimized-run-id",
        default=None,
        help="Run ID of an existing optimized run to compare (default: None, executes a new optimized run)",
    )

    args = parser.parse_args()

    try:
        compare_research_routing(
            routing_config=args.routing_config,
            baseline_model=args.model,
            cost_ceiling_usd=args.cost_ceiling,
            use_real_api=args.real,
            print_output=True,
            baseline_run_id=args.baseline_run_id,
            optimized_run_id=args.optimized_run_id,
        )
    except (ValueError, KeyError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error executing comparison: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

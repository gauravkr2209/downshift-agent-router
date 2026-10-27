"""End-to-End Reproducible Hackathon Demo for Downshift.

Executes the complete evidence-based routing lifecycle in a single command:
1. Baseline Agent Execution (Research Agent on Nemotron Ultra)
2. Step Replay across candidate parameter tiers (Super and Nano)
3. Objective Constraint Evaluation (Schema, JSON, Tool Call fidelity)
4. Semantic Quality Judging (Blinded LLM evaluation with pass/fail)
5. Routing Recommendations & Configuration Export (routing-test.json)
6. Dynamic Agent Execution under the exported routing configuration
7. Side-by-side Comparative Cost & Accuracy Verification

DISCLAIMER:
All runs executed in default mock mode represent simulated agent interactions.
They verify runtime routing mechanics, trace storage, and cost calculations.
They do not establish production model quality or verified real-world cost savings.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

# Ensure repository root is in sys.path when executed directly as a script
_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

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
from examples.research_agent.compare_runs import (
    SIMULATION_DISCLAIMER,
    compare_research_routing,
)
from server.app import db as db_mod
from server.app.checks import evaluate_run
from server.app.db import create_db_and_tables, sync_prices_from_yaml
from server.app.judge import judge_run
from server.app.models import Run, Step
from server.app.replay import MockReplayClient, replay_run
from server.app.router import export_routing_config, recommend_routing


def run_demo(
    export_path: str = "routing-test.json",
    cost_ceiling_usd: float = 0.20,
) -> dict[str, Any]:
    """Execute the complete Downshift optimization lifecycle sequentially."""
    create_db_and_tables()
    sync_prices_from_yaml()

    print("\n" + "=" * 80)
    print("           DOWNSHIFT: EVIDENCE-BASED MODEL ROUTING DEMO")
    print("=" * 80)
    print("Mode   : DETERMINISTIC SIMULATED MOCK")
    print(f"Notice : {SIMULATION_DISCLAIMER}")
    print("=" * 80 + "\n")

    # --------------------------------------------------------------------------
    # Step 1: Baseline Agent Execution
    # --------------------------------------------------------------------------
    print("[1/7] Running Baseline Research Agent on Nemotron Ultra (550B)...")
    mock_client = MockSearchAgentClient(model="nemotron_ultra")
    baseline_agent = ResearchAgent(
        client=mock_client,
        baseline_model="nemotron_ultra",
        cost_ceiling_usd=cost_ceiling_usd,
        routing_config=None,
    )
    baseline_res = baseline_agent.run(
        question=RESEARCH_QUESTION,
        checklist=CHECKLIST_ITEMS,
        max_searches=DEFAULT_MAX_SEARCHES,
    )
    baseline_run_id = baseline_res["run_id"]

    with Session(db_mod.engine) as session:
        b_run = session.get(Run, baseline_run_id)
        b_steps = session.exec(
            select(Step).where(Step.run_id == baseline_run_id).order_by(col(Step.idx))
        ).all()
        baseline_cost = b_run.total_cost_usd if b_run else 0.0

    print(f"  [OK] Baseline Run Recorded : {baseline_run_id}")
    print(f"  [OK] Execution Steps       : {len(b_steps)} steps")
    print(f"  [OK] Total Baseline Cost   : ${baseline_cost:.6f}")
    print(f"  [OK] Task Success          : {baseline_res['task_success']}")
    print(f"  [OK] Checklist Compliance  : {baseline_res['checklist_score']:.1%}\n")

    # --------------------------------------------------------------------------
    # Step 2: Step-Level Replay on Candidate Tiers
    # --------------------------------------------------------------------------
    print("[2/7] Replaying Steps across Candidate Tiers (Super: 120B, Nano: 30B)...")
    replay_client = MockReplayClient()
    replays = replay_run(
        run_id=baseline_run_id,
        tiers=["super", "nano"],
        client=replay_client,
        cost_ceiling_usd=cost_ceiling_usd,
        force=True,
    )
    total_replay_cost = sum(r.cost_usd for r in replays)
    print(f"  [OK] Replays Generated     : {len(replays)} candidate evaluations")
    print(f"  [OK] Replay Execution Cost : ${total_replay_cost:.6f}\n")

    # --------------------------------------------------------------------------
    # Step 3: Objective Constraint Evaluation
    # --------------------------------------------------------------------------
    print("[3/7] Running Objective Constraint Checks (JSON, Tool Calls, Format)...")
    with Session(db_mod.engine) as session:
        evaluations = evaluate_run(run_id=baseline_run_id, session=session)
        obj_passes = sum(1 for e in evaluations if e.objective_pass)
        obj_count = len(evaluations)
    obj_rate = (obj_passes / obj_count) if obj_count else 0.0
    print(f"  [OK] Objective Checks Run  : {obj_count}")
    print(f"  [OK] Hard Constraint Pass  : {obj_rate:.1%} ({obj_passes}/{obj_count})\n")

    # --------------------------------------------------------------------------
    # Step 4: Semantic Quality Judging
    # --------------------------------------------------------------------------
    print("[4/7] Running Blinded Semantic Quality Judge...")
    with Session(db_mod.engine) as session:
        judge_evals = judge_run(run_id=baseline_run_id, force=True, session=session)
        judge_passes = sum(
            1 for e in judge_evals if e.judge_reason and e.judge_reason.startswith("[PASS]")
        )
        judge_count = len(judge_evals)
    judge_rate = (judge_passes / judge_count) if judge_count else 0.0
    print(f"  [OK] Semantic Evals Judged : {judge_count}")
    print(f"  [OK] Semantic Pass Rate    : {judge_rate:.1%} ({judge_passes}/{judge_count})\n")

    # --------------------------------------------------------------------------
    # Step 5: Routing Recommendations & Config Export
    # --------------------------------------------------------------------------
    print("[5/7] Generating Evidence-Based Routing Recommendations...")
    report = recommend_routing(run_id=baseline_run_id, min_pass_rate=0.95, min_samples=3)
    export_routing_config(report, export_path)

    print("  [OK] Recommendations by Tag:")
    for tag, rec in report.recommendations.items():
        print(f"    - {tag:<16} -> {rec.recommended_tier:<8} ({rec.decision_reason})")
    print(
        f"  [OK] Estimated Savings     : {report.estimated_savings_pct:.1f}% (${report.estimated_savings_usd:.6f})"
    )
    print(f"  [OK] Exported Routing Map  : {export_path}\n")

    # --------------------------------------------------------------------------
    # Step 6: Dynamic Agent Run with Exported Routing Config
    # --------------------------------------------------------------------------
    print(f"[6/7] Executing Agent with Dynamic Routing ({export_path})...")
    opt_client = MockSearchAgentClient(model="nemotron_ultra")
    opt_agent = ResearchAgent(
        client=opt_client,
        baseline_model="nemotron_ultra",
        cost_ceiling_usd=cost_ceiling_usd,
        routing_config=export_path,
    )
    opt_res = opt_agent.run(
        question=RESEARCH_QUESTION,
        checklist=CHECKLIST_ITEMS,
        max_searches=DEFAULT_MAX_SEARCHES,
    )
    opt_run_id = opt_res["run_id"]

    with Session(db_mod.engine) as session:
        o_run = session.get(Run, opt_run_id)
        o_steps = session.exec(
            select(Step).where(Step.run_id == opt_run_id).order_by(col(Step.idx))
        ).all()
        opt_cost = o_run.total_cost_usd if o_run else 0.0

    print(f"  [OK] Optimized Run Recorded: {opt_run_id}")
    print(f"  [OK] Execution Steps       : {len(o_steps)} steps")
    print(f"  [OK] Total Optimized Cost  : ${opt_cost:.6f}")
    print(f"  [OK] Task Success          : {opt_res['task_success']}")
    print(f"  [OK] Checklist Compliance  : {opt_res['checklist_score']:.1%}\n")

    # --------------------------------------------------------------------------
    # Step 7: Side-by-Side Comparison Verification
    # --------------------------------------------------------------------------
    print("[7/7] Verifying Comparative Savings and Step Selection...")
    comparison = compare_research_routing(
        routing_config=export_path,
        baseline_model="nemotron_ultra",
        cost_ceiling_usd=cost_ceiling_usd,
        use_real_api=False,
        print_output=True,
        baseline_run_id=baseline_run_id,
        optimized_run_id=opt_run_id,
    )

    print("=" * 80)
    print("                    DEMO EXECUTION COMPLETE")
    print("=" * 80)
    print(f"Baseline Run ID  : {baseline_run_id}")
    print(f"Optimized Run ID : {opt_run_id}")
    print(f"Exported Config  : {export_path}")
    print(f"Simulated Savings: {comparison['savings_pct']:.1f}% (${comparison['savings_usd']:.6f})")
    print(f"Notice           : {SIMULATION_DISCLAIMER}")
    print("=" * 80 + "\n")

    return {
        "baseline_run_id": baseline_run_id,
        "optimized_run_id": opt_run_id,
        "export_path": export_path,
        "baseline_cost_usd": baseline_cost,
        "optimized_cost_usd": opt_cost,
        "savings_usd": comparison["savings_usd"],
        "savings_pct": comparison["savings_pct"],
        "step_comparisons": comparison["step_comparisons"],
        "disclaimer": SIMULATION_DISCLAIMER,
    }


def main() -> None:
    """CLI entrypoint for running the end-to-end demo."""
    parser = argparse.ArgumentParser(
        description="Run end-to-end reproducible hackathon demo of Downshift"
    )
    parser.add_argument(
        "--export",
        default="routing-test.json",
        help="Export path for routing configuration JSON (default: routing-test.json)",
    )
    parser.add_argument(
        "--cost-ceiling",
        type=float,
        default=0.20,
        help="Per-run cost ceiling in USD (default: 0.20)",
    )

    args = parser.parse_args()

    try:
        run_demo(
            export_path=args.export,
            cost_ceiling_usd=args.cost_ceiling,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error during demo execution: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

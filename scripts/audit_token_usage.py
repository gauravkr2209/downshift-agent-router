"""Read-only local database audit for token usage and estimated costs.

Inspects Downshift's SQLite database to audit recorded model steps, candidate
replays, token usage, and client-side estimated costs without making any
network or inference requests.
"""

from pathlib import Path
from typing import Any

from sqlalchemy import inspect
from sqlmodel import Session, col, select

from server.app import db as db_mod
from server.app.config import REPO_ROOT, settings
from server.app.models import Replay, Run, Step


def format_usd(val: float) -> str:
    """Format floating point USD amount."""
    return f"${val:.6f}"


def classify_tier(model_id: str) -> str:
    """Classify model ID into Ultra, Super, Nano, or Other."""
    m = model_id.lower()
    if "ultra" in m or "550b" in m:
        return "ultra"
    if "super" in m or "120b" in m:
        return "super"
    if "nano" in m or "30b" in m or "lightning" in m:
        return "nano"
    return "other"


def is_step_usage_missing(step: Step) -> bool:
    """Determine whether a step is missing usage metadata."""
    if step.kind == "error":
        return True
    if step.tokens_in == 0 and step.tokens_out == 0:
        resp = step.response
        if resp is None:
            return True
        if isinstance(resp, dict):
            usage = resp.get("usage")
            if usage is None or not isinstance(usage, dict):
                return True
            if usage.get("prompt_tokens") is None or usage.get("completion_tokens") is None:
                return True
    return False


def is_replay_usage_missing(replay: Replay) -> bool:
    """Determine whether a replay candidate is missing usage metadata."""
    if replay.error is not None:
        return True
    if replay.tokens_in == 0 and replay.tokens_out == 0:
        resp = replay.response
        if resp is None:
            return True
        if isinstance(resp, dict):
            usage = resp.get("usage")
            if usage is None or not isinstance(usage, dict):
                return True
            if usage.get("prompt_tokens") is None or usage.get("completion_tokens") is None:
                return True
    return False


def run_token_audit(session: Session | None = None) -> dict[str, Any]:
    """Execute read-only token and cost audit across database tables.

    Returns audit metrics dictionary and prints formatted diagnostic report.
    This function is strictly read-only: it will not initialize or mutate a database.
    """
    close_session = False
    if session is None:
        db_url = settings.database_url
        if db_url.startswith("sqlite:///"):
            db_target = db_url.replace("sqlite:///", "")
            if db_target != ":memory:":
                target_file = Path(db_target)
                if not target_file.is_absolute():
                    target_file = REPO_ROOT / target_file
                if not target_file.exists():
                    raise FileNotFoundError(
                        f"Target database '{target_file}' does not exist. "
                        "The token audit is strictly read-only and will not create or initialize a new database."
                    )
        session = Session(db_mod.engine)
        close_session = True

    try:
        # Check if tables exist before querying; do NOT create or mutate tables
        inspector = inspect(session.connection())
        table_names = set(inspector.get_table_names())
        if not {"runs", "steps", "replays"}.issubset(table_names):
            print("=" * 80)
            print("                 DOWNSHIFT: TOKEN USAGE & ESTIMATED COST AUDIT")
            print("=" * 80)
            print("Data Source : Local database")
            print("Network     : ZERO remote or inference requests dispatched (Read-Only)")
            print("=" * 80)
            print("\n[NOTICE] Database is currently empty.")
            print("  - Recorded runs     : 0")
            print("  - Recorded steps    : 0")
            print("  - Candidate replays : 0")
            print("\nSchema Disclosures:")
            print(
                "  - Available in DB   : run_id, task_name, model, prompt_tokens, completion_tokens, cost_usd"
            )
            print("  - UNAVAILABLE in DB : actual provider invoiced billing, prompt cache metrics")
            print("=" * 80)
            return {
                "runs_count": 0,
                "steps_count": 0,
                "replays_count": 0,
                "step_tokens_in": 0,
                "step_tokens_out": 0,
                "step_total_tokens": 0,
                "step_total_cost_usd": 0.0,
                "step_reconciled_cost_usd": 0.0,
                "missing_usage_steps": 0,
                "has_unresolved_step_costs": False,
                "replay_total_tokens": 0,
                "replay_total_cost_usd": 0.0,
                "replay_reconciled_cost_usd": 0.0,
                "missing_usage_replays": 0,
                "has_unresolved_replay_costs": False,
            }

        runs = session.exec(select(Run).order_by(col(Run.started_at).desc())).all()
        steps = session.exec(select(Step).order_by(col(Step.created_at).desc())).all()
        replays = session.exec(select(Replay).order_by(col(Replay.created_at).desc())).all()

        print("=" * 80)
        print("                 DOWNSHIFT: TOKEN USAGE & ESTIMATED COST AUDIT")
        print("=" * 80)
        print("Data Source : Local database (downshift.db)")
        print("Network     : ZERO remote or inference requests dispatched (Read-Only)")
        print("=" * 80)

        # Handle empty database
        if not runs and not steps and not replays:
            print("\n[NOTICE] Database is currently empty.")
            print("  - Recorded runs     : 0")
            print("  - Recorded steps    : 0")
            print("  - Candidate replays : 0")
            print("\nSchema Disclosures:")
            print(
                "  - Available in DB   : run_id, task_name, model, prompt_tokens, completion_tokens, cost_usd"
            )
            print("  - UNAVAILABLE in DB : actual provider invoiced billing, prompt cache metrics")
            print("=" * 80)
            return {
                "runs_count": 0,
                "steps_count": 0,
                "replays_count": 0,
                "step_tokens_in": 0,
                "step_tokens_out": 0,
                "step_total_tokens": 0,
                "step_total_cost_usd": 0.0,
                "step_reconciled_cost_usd": 0.0,
                "missing_usage_steps": 0,
                "has_unresolved_step_costs": False,
                "replay_total_tokens": 0,
                "replay_total_cost_usd": 0.0,
                "replay_reconciled_cost_usd": 0.0,
                "missing_usage_replays": 0,
                "has_unresolved_replay_costs": False,
            }

        # 1. Primary Agent Steps Summary
        step_tokens_in = 0
        step_tokens_out = 0
        step_cost_sum = 0.0
        step_reconciled_cost = 0.0
        missing_usage_steps = 0

        # Model breakdown
        model_stats: dict[str, dict[str, Any]] = {}
        # Tag breakdown
        tag_stats: dict[str, dict[str, Any]] = {}

        for s in steps:
            s_missing = is_step_usage_missing(s)
            if s_missing:
                missing_usage_steps += 1
            else:
                step_tokens_in += s.tokens_in
                step_tokens_out += s.tokens_out
                step_reconciled_cost += s.cost_usd
            step_cost_sum += s.cost_usd

            # Group by exact model ID
            m_key = s.model
            if m_key not in model_stats:
                model_stats[m_key] = {
                    "tier": classify_tier(m_key),
                    "steps": 0,
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "reconciled_cost_usd": 0.0,
                    "missing_usage": 0,
                }
            model_stats[m_key]["steps"] += 1
            if s_missing:
                model_stats[m_key]["missing_usage"] += 1
            else:
                model_stats[m_key]["tokens_in"] += s.tokens_in
                model_stats[m_key]["tokens_out"] += s.tokens_out
                model_stats[m_key]["reconciled_cost_usd"] += s.cost_usd

            # Group by step tag
            t_key = s.tag
            if t_key not in tag_stats:
                tag_stats[t_key] = {
                    "steps": 0,
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "reconciled_cost_usd": 0.0,
                    "missing_usage": 0,
                }
            tag_stats[t_key]["steps"] += 1
            if s_missing:
                tag_stats[t_key]["missing_usage"] += 1
            else:
                tag_stats[t_key]["tokens_in"] += s.tokens_in
                tag_stats[t_key]["tokens_out"] += s.tokens_out
                tag_stats[t_key]["reconciled_cost_usd"] += s.cost_usd

        step_total_tokens = step_tokens_in + step_tokens_out

        # 2. Candidate Replays Summary (Kept strictly separated to prevent double-counting)
        replay_tokens_in = 0
        replay_tokens_out = 0
        replay_cost_sum = 0.0
        replay_reconciled_cost = 0.0
        missing_usage_replays = 0

        for r in replays:
            r_missing = is_replay_usage_missing(r)
            if r_missing:
                missing_usage_replays += 1
            else:
                replay_tokens_in += r.tokens_in
                replay_tokens_out += r.tokens_out
                replay_reconciled_cost += r.cost_usd
            replay_cost_sum += r.cost_usd

        replay_total_tokens = replay_tokens_in + replay_tokens_out

        # Print Executive Summary
        print("\n--- 1. OVERALL TOTALS & INTEGRITY CHECK ---")
        print(f"Total Agent Runs Recorded   : {len(runs)}")
        print(f"Total Agent Execution Steps : {len(steps)}")
        print(f"  - Reconciled Steps        : {len(steps) - missing_usage_steps}")
        print(
            f"  - Missing-Usage Steps     : {missing_usage_steps} (excluded from token sums; cost is UNRESOLVED)"
        )
        print(f"Input (Prompt) Tokens       : {step_tokens_in:,}")
        print(f"Output (Completion) Tokens  : {step_tokens_out:,}")
        print(f"Total Reconciled Tokens     : {step_total_tokens:,}")
        if missing_usage_steps > 0:
            print(
                f"Reconciled Estimated Cost   : {format_usd(step_reconciled_cost)} [INCOMPLETE - {missing_usage_steps} step(s) unresolved]"
            )
            print(
                f"Unresolved Steps Cost       : UNKNOWN ({missing_usage_steps} call(s) timed out or missing usage; NOT confirmed $0.00)"
            )
            print(
                f"Recorded Cost Subtotal      : {format_usd(step_cost_sum)} [PARTIAL ONLY - provider may have billed unresolved calls]"
            )
        else:
            print(f"Reconciled Estimated Cost   : {format_usd(step_reconciled_cost)} [COMPLETE]")
            print(f"Recorded Estimated Cost     : {format_usd(step_cost_sum)}")

        replay_note = (
            f" [INCOMPLETE: {missing_usage_replays} unrec replay(s)]"
            if missing_usage_replays > 0
            else ""
        )
        print(
            f"Candidate Replay Calls      : {len(replays)} (Tokens: {replay_total_tokens:,} | Reconciled Cost: {format_usd(replay_reconciled_cost)}{replay_note})"
        )
        if missing_usage_replays > 0:
            print(
                f"  - Unresolved Replays Cost : UNKNOWN ({missing_usage_replays} replay(s) failed or timed out; NOT confirmed $0.00)"
            )
        print(
            "  [NOTE: Replay calls are tracked separately and NOT double-counted into run totals]"
        )

        # Print Model Breakdown
        print("\n--- 2. USAGE & ESTIMATED COST BY MODEL ID & TIER ---")
        print(
            f"{'Tier':<8} {'Model Identifier':<42} {'Steps':>6} {'Prompt Tok':>12} {'Compl Tok':>12} {'Reconciled Cost':>18} {'Unresolved':>12}"
        )
        print("-" * 114)
        for m_id, m_data in sorted(model_stats.items(), key=lambda x: x[1]["tier"]):
            cost_flag = "*" if m_data["missing_usage"] > 0 else " "
            cost_str = f"{format_usd(m_data['reconciled_cost_usd'])}{cost_flag}"
            print(
                f"{m_data['tier'].upper():<8} "
                f"{m_id:<42} "
                f"{m_data['steps']:>6} "
                f"{m_data['tokens_in']:>12,}"
                f"{m_data['tokens_out']:>12,}"
                f"{cost_str:>18} "
                f"{m_data['missing_usage']:>12}"
            )
        if any(m_data["missing_usage"] > 0 for m_data in model_stats.values()):
            print(
                "  * Note: Reconciled cost is partial; excludes unresolved/timed-out call(s) with unknown cost."
            )

        # Print Tag Breakdown
        print("\n--- 3. USAGE & ESTIMATED COST BY STEP TAG ---")
        print(
            f"{'Step Tag':<25} {'Steps':>6} {'Prompt Tok':>12} {'Compl Tok':>12} {'Total Tok':>12} {'Reconciled Cost':>18} {'Unresolved':>12}"
        )
        print("-" * 103)
        for t_name, t_data in sorted(tag_stats.items(), key=lambda x: x[0]):
            tot_tok = t_data["tokens_in"] + t_data["tokens_out"]
            cost_flag = "*" if t_data["missing_usage"] > 0 else " "
            cost_str = f"{format_usd(t_data['reconciled_cost_usd'])}{cost_flag}"
            print(
                f"{t_name:<25} "
                f"{t_data['steps']:>6} "
                f"{t_data['tokens_in']:>12,} "
                f"{t_data['tokens_out']:>12,} "
                f"{tot_tok:>12,} "
                f"{cost_str:>18} "
                f"{t_data['missing_usage']:>12}"
            )
        if any(t_data["missing_usage"] > 0 for t_data in tag_stats.values()):
            print(
                "  * Note: Reconciled cost is partial; excludes unresolved/timed-out call(s) with unknown cost."
            )

        # Print Runs List
        print("\n--- 4. RECORDED AGENT RUNS ---")
        print(
            f"{'Started At (UTC)':<20} {'Run ID':<38} {'Task Name':<20} {'Status':<15} {'Steps':>5} {'Tokens':>10} {'Est Cost':>14}"
        )
        print("-" * 128)
        for run_item in runs:
            r_tokens = run_item.tokens_in + run_item.tokens_out
            unreconciled_flag = (
                " (UNREC)" if (run_item.meta or {}).get("has_unreconciled_cost") else ""
            )
            cost_note = "*" if (run_item.meta or {}).get("has_unreconciled_cost") else ""
            status_display = f"{run_item.status}{unreconciled_flag}"
            cost_display = f"{format_usd(run_item.total_cost_usd)}{cost_note}"
            print(
                f"{run_item.started_at.strftime('%Y-%m-%d %H:%M:%S'):<20} "
                f"{run_item.id:<38} "
                f"{run_item.task_name[:18]:<20} "
                f"{status_display:<15} "
                f"{len(run_item.steps):>5} "
                f"{r_tokens:>10,} "
                f"{cost_display:>14}"
            )
        if any((r.meta or {}).get("has_unreconciled_cost") for r in runs):
            print(
                "  * Note: Run has unresolved costs from timed-out or missing-usage steps; total cost is incomplete."
            )

        # Disclosures and Caveats
        print("\n" + "=" * 80)
        print("AUDIT METHODOLOGY & FIELD PROVENANCE:")
        print("  1. Token Counts: Sourced directly from provider 'usage' metadata in responses.")
        print("  2. Cost (USD): Calculated locally using pricing tables in prices.yaml.")
        print("  3. Provider Invoiced Billing: UNAVAILABLE in local database.")
        print("     (OpenAI-compatible APIs do not return billed dollar amounts).")
        print(
            "  4. Missing Usage & Timeouts: Recorded steps with absent or timed-out usage are NOT"
        )
        print("     treated as zero consumption; they are tracked as 'UNRESOLVED' and flagged.")
        print("     A recorded $0.00 for an error step does NOT mean the call was free.")
        print("  5. Anti-Double-Counting: Primary agent steps and offline candidate replays are")
        print("     partitioned in separate database tables and aggregated independently.")
        print("=" * 80)

        return {
            "runs_count": len(runs),
            "steps_count": len(steps),
            "replays_count": len(replays),
            "step_tokens_in": step_tokens_in,
            "step_tokens_out": step_tokens_out,
            "step_total_tokens": step_total_tokens,
            "step_total_cost_usd": step_cost_sum,
            "step_reconciled_cost_usd": step_reconciled_cost,
            "missing_usage_steps": missing_usage_steps,
            "has_unresolved_step_costs": missing_usage_steps > 0,
            "replay_total_tokens": replay_total_tokens,
            "replay_total_cost_usd": replay_cost_sum,
            "replay_reconciled_cost_usd": replay_reconciled_cost,
            "missing_usage_replays": missing_usage_replays,
            "has_unresolved_replay_costs": missing_usage_replays > 0,
        }
    finally:
        if close_session:
            session.close()


if __name__ == "__main__":
    run_token_audit()

"""Command-line interface to inspect Downshift runs and verify cost integrity."""

import argparse
import sys
from typing import Any

from sqlmodel import Session, col, select

from server.app import db as db_mod
from server.app.checks import evaluate_run
from server.app.config import settings
from server.app.db import create_db_and_tables, sync_prices_from_yaml
from server.app.judge import MockJudgeClient, judge_run
from server.app.models import Evaluation, Replay, Run, Step
from server.app.replay import MockReplayClient, replay_run
from server.app.router import export_routing_config, recommend_routing


def format_usd(val: float) -> str:
    """Format floating point USD amount."""
    return f"${val:.6f}"


def resolve_run_id(run_id: str, session: Session) -> str:
    """Resolve a run identifier, mapping 'latest' to the most recent run ID."""
    clean = run_id.strip()
    if clean.lower() == "latest":
        latest_run = session.exec(select(Run).order_by(col(Run.started_at).desc())).first()
        if not latest_run:
            print("Error: No recorded runs found in database.", file=sys.stderr)
            sys.exit(1)
        return latest_run.id
    return clean


def inspect_run(run_id: str, session: Session | None = None) -> bool:
    """Print run details, step count, total cost, and verify total equals sum of step costs."""
    create_db_and_tables()
    try:
        sync_prices_from_yaml()
    except (FileNotFoundError, ValueError):
        pass
    close_session = False
    if session is None:
        session = Session(db_mod.engine)
        close_session = True

    try:
        target_run_id = resolve_run_id(run_id, session)
        run_record = session.get(Run, target_run_id)
        if not run_record:
            print(f"Error: Run '{run_id}' not found.", file=sys.stderr)
            return False
        run_id = target_run_id

        steps = session.exec(
            select(Step).where(Step.run_id == run_id).order_by(col(Step.idx))
        ).all()

        step_cost_sum = sum(s.cost_usd for s in steps)
        step_tokens_in = sum(s.tokens_in for s in steps)
        step_tokens_out = sum(s.tokens_out for s in steps)

        is_cost_verified = abs(run_record.total_cost_usd - step_cost_sum) < 1e-9
        is_tokens_verified = (
            run_record.tokens_in == step_tokens_in and run_record.tokens_out == step_tokens_out
        )

        print("=" * 60)
        print(f"Downshift Run Inspection: {run_record.id}")
        print("=" * 60)
        print(f"Task Name      : {run_record.task_name}")
        print(f"Agent Name     : {run_record.agent_name}")
        print(f"Baseline Model : {run_record.baseline_model}")
        print(f"Status         : {run_record.status}")
        print(f"Started At     : {run_record.started_at.isoformat()}")
        print(f"Step Count     : {len(steps)}")
        print(f"Tokens In      : {run_record.tokens_in} (Sum: {step_tokens_in})")
        print(f"Tokens Out     : {run_record.tokens_out} (Sum: {step_tokens_out})")
        print(f"Total Cost     : {format_usd(run_record.total_cost_usd)}")
        print(f"Sum Step Costs : {format_usd(step_cost_sum)}")
        print("-" * 60)

        if steps:
            print("Steps Breakdown:")
            for s in steps:
                print(
                    f"  [{s.idx:02d}] Tag: {s.tag:<15} Kind: {s.kind:<10} "
                    f"Tokens: {s.tokens_in:>4} in / {s.tokens_out:>4} out | Cost: {format_usd(s.cost_usd)}"
                )
            print("-" * 60)

        if is_cost_verified and is_tokens_verified:
            print("[VERIFIED] Total cost and token totals exactly match sum of step records.")
            return True
        else:
            print(
                "[INTEGRITY ERROR] Discrepancy detected between run total and step sums!",
                file=sys.stderr,
            )
            return False
    finally:
        if close_session:
            session.close()


def list_runs(session: Session | None = None) -> None:
    """List recent runs recorded in the database."""
    create_db_and_tables()
    try:
        sync_prices_from_yaml()
    except (FileNotFoundError, ValueError):
        pass
    close_session = False
    if session is None:
        session = Session(db_mod.engine)
        close_session = True

    try:
        runs = session.exec(select(Run).order_by(col(Run.started_at).desc())).all()
        if not runs:
            print("No recorded runs found in database.")
            return

        print(f"{'Run ID':<38} {'Task':<16} {'Agent':<14} {'Status':<10} {'Cost':<12}")
        print("-" * 92)
        for r in runs:
            print(
                f"{r.id:<38} {r.task_name:<16} {r.agent_name:<14} {r.status:<10} {format_usd(r.total_cost_usd):<12}"
            )
    finally:
        if close_session:
            session.close()


def replay_command(args: argparse.Namespace) -> None:
    """Execute replay subcommand."""
    tier_list = [t.strip() for t in args.tiers.split(",") if t.strip()]
    if not tier_list:
        print("Error: No tiers specified.", file=sys.stderr)
        sys.exit(1)

    create_db_and_tables()
    with Session(db_mod.engine) as session:
        args.run_id = resolve_run_id(args.run_id, session)

    if args.real:
        if (
            not settings.nebius_api_key
            or "PLACEHOLDER" in settings.nebius_api_key.upper()
            or settings.nebius_api_key.startswith("your_")
        ):
            print("REFUSAL: NEBIUS_API_KEY is not configured in .env.", file=sys.stderr)
            sys.exit(1)

        if not settings.nebius_base_url or "PLACEHOLDER" in settings.nebius_base_url.upper():
            print("REFUSAL: NEBIUS_BASE_URL is not configured in .env.", file=sys.stderr)
            sys.exit(1)

        is_verified, reason = db_mod.check_live_pricing_verified()
        if not is_verified:
            print(f"REFUSAL: {reason}", file=sys.stderr)
            sys.exit(1)

        try:
            sync_prices_from_yaml()
        except (FileNotFoundError, ValueError) as exc:
            print(f"REFUSAL: Pricing synchronization failed: {exc}", file=sys.stderr)
            sys.exit(1)

        from openai import OpenAI

        client: Any = OpenAI(
            api_key=settings.nebius_api_key,
            base_url=settings.nebius_base_url,
            timeout=settings.nebius_request_timeout,
            max_retries=0,
        )
        max_workers = 1
        retry_attempts = 1
        is_live = True
        print(
            f"Replaying run '{args.run_id}' across tiers: {', '.join(tier_list)} (Mode: LIVE - Nebius Token Factory)"
        )
    else:
        client = MockReplayClient()
        max_workers = args.max_workers
        retry_attempts = None
        is_live = False
        print(f"Replaying run '{args.run_id}' across tiers: {', '.join(tier_list)} (Mode: MOCK)")

    try:
        replays = replay_run(
            run_id=args.run_id,
            tiers=tier_list,
            client=client,
            max_workers=max_workers,
            max_steps=args.max_steps,
            cost_ceiling_usd=args.cost_ceiling,
            force=args.force,
            retry_attempts=retry_attempts,
            is_live=is_live,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Error during replay: {exc}", file=sys.stderr)
        sys.exit(1)

    total_cost = sum(r.cost_usd for r in replays)
    print("=" * 80)
    print(f"Downshift Replay Summary: Run {args.run_id}")
    print("=" * 80)
    print(
        f"{'Tier':<10} {'Status':<10} {'Tokens In':<12} {'Tokens Out':<12} {'Cost':<12} {'Latency':<10}"
    )
    print("-" * 80)
    for r in replays:
        status_label = "ERROR" if r.error else "OK"
        print(
            f"{r.tier:<10} {status_label:<10} {r.tokens_in:<12} {r.tokens_out:<12} "
            f"{format_usd(r.cost_usd):<12} {r.latency_ms:>6.1f}ms"
        )
    print("-" * 80)
    print(f"Total Replays Generated : {len(replays)}")
    print(f"Total Replay Cost       : {format_usd(total_cost)}")


def evaluate_command(run_id: str, session: Session | None = None) -> None:
    """Run objective evaluation on all replays for a run and print pass rate per tag and tier."""
    create_db_and_tables()
    close_session = False
    if session is None:
        session = Session(db_mod.engine)
        close_session = True

    try:
        target_run_id = resolve_run_id(run_id, session)
        run_record = session.get(Run, target_run_id)
        if not run_record:
            print(f"Error: Run '{run_id}' not found.", file=sys.stderr)
            sys.exit(1)
        run_id = target_run_id

        evaluations = evaluate_run(run_id, session=session)

        steps = session.exec(
            select(Step).where(Step.run_id == run_id).order_by(col(Step.idx))
        ).all()

        tag_tier_stats: dict[tuple[str, str], list[int]] = {}
        tier_stats: dict[str, list[int]] = {}

        for step in steps:
            replays = session.exec(
                select(Replay).where(Replay.step_id == step.id).order_by(col(Replay.tier))
            ).all()
            for r in replays:
                ev = session.exec(select(Evaluation).where(Evaluation.replay_id == r.id)).first()
                is_pass = ev.objective_pass if ev else False

                key = (step.tag, r.tier)
                if key not in tag_tier_stats:
                    tag_tier_stats[key] = [0, 0]
                tag_tier_stats[key][1] += 1
                if is_pass:
                    tag_tier_stats[key][0] += 1

                if r.tier not in tier_stats:
                    tier_stats[r.tier] = [0, 0]
                tier_stats[r.tier][1] += 1
                if is_pass:
                    tier_stats[r.tier][0] += 1

        print("=" * 80)
        print(f"Downshift Objective Evaluation Summary: Run {run_id}")
        print("=" * 80)
        print(f"{'Tag':<25} {'Tier':<12} {'Pass Rate':<12} {'Passed / Total':<15}")
        print("-" * 80)
        for (tag, tier), (passed, total) in tag_tier_stats.items():
            rate_str = f"{(passed / total) * 100:.1f}%" if total > 0 else "0.0%"
            print(f"{tag:<25} {tier:<12} {rate_str:<12} {passed:>3} / {total:<3}")

        print("-" * 80)
        print("Overall Pass Rate by Tier:")
        for tier, (passed, total) in tier_stats.items():
            rate_str = f"{(passed / total) * 100:.1f}%" if total > 0 else "0.0%"
            print(f"  {tier:<10}: {rate_str:<7} ({passed} / {total})")
        print("=" * 80)
        print(f"Total Evaluations Recorded: {len(evaluations)}")

        if not tag_tier_stats and len(evaluations) == 0:
            print(
                f"Notice: No replays found for run '{run_id}'. "
                f"Run 'python -m server.app.cli replay {run_id} --tiers super,nano' first.",
                file=sys.stderr,
            )
    finally:
        if close_session:
            session.close()


def judge_command(args: argparse.Namespace, session: Session | None = None) -> None:
    """Execute LLM quality judging for a run and print pass rate and quality report."""
    create_db_and_tables()
    close_session = False
    if session is None:
        session = Session(db_mod.engine)
        close_session = True

    try:
        target_run_id = resolve_run_id(args.run_id, session)
        run_record = session.get(Run, target_run_id)
        if not run_record:
            print(f"Error: Run '{args.run_id}' not found.", file=sys.stderr)
            sys.exit(1)
        args.run_id = target_run_id

        is_live = bool(getattr(args, "real", False))
        cost_ceiling = getattr(args, "cost_ceiling", 0.07)
        if is_live:
            if (
                not settings.nebius_api_key
                or "PLACEHOLDER" in settings.nebius_api_key.upper()
                or settings.nebius_api_key.startswith("your_")
            ):
                print("REFUSAL: NEBIUS_API_KEY is not configured in .env.", file=sys.stderr)
                sys.exit(1)

            if not settings.nebius_base_url or "PLACEHOLDER" in settings.nebius_base_url.upper():
                print("REFUSAL: NEBIUS_BASE_URL is not configured in .env.", file=sys.stderr)
                sys.exit(1)

            is_verified, reason = db_mod.check_live_pricing_verified()
            if not is_verified:
                print(f"REFUSAL: {reason}", file=sys.stderr)
                sys.exit(1)

            try:
                sync_prices_from_yaml()
            except (FileNotFoundError, ValueError) as exc:
                print(f"REFUSAL: Pricing synchronization failed: {exc}", file=sys.stderr)
                sys.exit(1)

            from openai import OpenAI

            client: Any = OpenAI(
                api_key=settings.nebius_api_key,
                base_url=settings.nebius_base_url,
                timeout=settings.nebius_request_timeout,
                max_retries=0,
            )
            print(
                f"Judging run '{args.run_id}' (Mode: LIVE - Nebius Token Factory Judge) [Budget Ceiling: ${cost_ceiling:.4f}]"
            )
        else:
            client = MockJudgeClient()
            print(f"Judging run '{args.run_id}' (Mode: MOCK Judge)")

        try:
            evaluations = judge_run(
                run_id=args.run_id,
                client=client,
                judge_model=getattr(args, "model", None),
                force=getattr(args, "force", False),
                session=session,
                is_live=is_live,
                cost_ceiling_usd=cost_ceiling,
            )
        except (ValueError, RuntimeError, OSError) as exc:
            print(f"Error during judging: {exc}", file=sys.stderr)
            sys.exit(1)

        steps = session.exec(
            select(Step).where(Step.run_id == args.run_id).order_by(col(Step.idx))
        ).all()

        tag_tier_stats: dict[tuple[str, str], list[Any]] = {}
        tier_stats: dict[str, list[Any]] = {}

        for step in steps:
            replays = session.exec(
                select(Replay).where(Replay.step_id == step.id).order_by(col(Replay.tier))
            ).all()
            for r in replays:
                ev = session.exec(select(Evaluation).where(Evaluation.replay_id == r.id)).first()
                if not ev:
                    continue

                is_pass = ev.judge_reason.startswith("[PASS]") if ev.judge_reason else False
                score = ev.judge_score

                key = (step.tag, r.tier)
                if key not in tag_tier_stats:
                    tag_tier_stats[key] = [0, 0, []]
                tag_tier_stats[key][1] += 1
                if is_pass:
                    tag_tier_stats[key][0] += 1
                if score is not None:
                    tag_tier_stats[key][2].append(score)

                if r.tier not in tier_stats:
                    tier_stats[r.tier] = [0, 0, []]
                tier_stats[r.tier][1] += 1
                if is_pass:
                    tier_stats[r.tier][0] += 1
                if score is not None:
                    tier_stats[r.tier][2].append(score)

        print("=" * 80)
        print(f"Downshift Replay Quality Report: Run {args.run_id}")
        print("=" * 80)
        print(
            f"{'Tag':<25} {'Tier':<10} {'Pass Rate':<12} {'Avg Score':<12} {'Passed / Total':<15}"
        )
        print("-" * 80)
        for (tag, tier), (passed, total, scores) in tag_tier_stats.items():
            rate_str = f"{(passed / total) * 100:.1f}%" if total > 0 else "0.0%"
            avg_str = f"{sum(scores) / len(scores):.2f}" if scores else "N/A"
            print(f"{tag:<25} {tier:<10} {rate_str:<12} {avg_str:<12} {passed:>3} / {total:<3}")

        print("-" * 80)
        print("Overall Judge Pass Rate by Tier:")
        for tier, (passed, total, scores) in tier_stats.items():
            rate_str = f"{(passed / total) * 100:.1f}%" if total > 0 else "0.0%"
            avg_str = f"{sum(scores) / len(scores):.2f}" if scores else "N/A"
            print(f"  {tier:<10}: {rate_str:<7} ({passed} / {total}) | Avg Score: {avg_str}")
        print("=" * 80)
        print(f"Total Evaluations Judged: {len(evaluations)}")
        total_judge_cost = sum(ev.judge_cost_usd or 0.0 for ev in evaluations)
        total_judge_tokens_in = sum(ev.judge_tokens_in or 0 for ev in evaluations)
        total_judge_tokens_out = sum(ev.judge_tokens_out or 0 for ev in evaluations)
        if total_judge_cost > 0.0 or total_judge_tokens_in > 0:
            print(
                f"Total Judge Tokens      : {total_judge_tokens_in} in / {total_judge_tokens_out} out"
            )
            print(f"Total Judge Cost        : {format_usd(total_judge_cost)}")

        if not tag_tier_stats and len(evaluations) == 0:
            print(
                f"Notice: No evaluated replays found for run '{args.run_id}'. "
                f"Run 'python -m server.app.cli replay {args.run_id}' and 'evaluate' first.",
                file=sys.stderr,
            )
    finally:
        if close_session:
            session.close()


def recommend_command(args: argparse.Namespace, session: Session | None = None) -> None:
    """Compute and display model routing recommendations for a run."""
    create_db_and_tables()
    close_session = False
    if session is None:
        session = Session(db_mod.engine)
        close_session = True

    try:
        target_run_id = resolve_run_id(args.run_id, session)
        run_record = session.get(Run, target_run_id)
        if not run_record:
            print(f"Error: Run '{args.run_id}' not found.", file=sys.stderr)
            sys.exit(1)
        args.run_id = target_run_id

        min_pass_rate = getattr(args, "min_pass_rate", 0.95)
        min_samples = getattr(args, "min_samples", 3)
        if not (0.0 <= min_pass_rate <= 1.0):
            print(
                f"Error: --min-pass-rate must be between 0.0 and 1.0, got {min_pass_rate}.",
                file=sys.stderr,
            )
            sys.exit(1)
        if min_samples < 1:
            print(
                f"Error: --min-samples must be at least 1, got {min_samples}.",
                file=sys.stderr,
            )
            sys.exit(1)

        report = recommend_routing(
            run_id=args.run_id,
            min_pass_rate=min_pass_rate,
            min_samples=min_samples,
            session=session,
        )

        if getattr(args, "export", None):
            export_routing_config(report, args.export)
            print(f"Exported routing map to: {args.export}", file=sys.stderr)

        if getattr(args, "json", False):
            print(report.model_dump_json(indent=2))
            return

        print("=" * 80)
        print(f"Downshift Model Routing Recommendations: Run {args.run_id}")
        print(
            f"Threshold: {report.min_pass_rate * 100:.1f}% pass rate | Min samples: {report.min_samples}"
        )
        print("=" * 80)
        print(
            f"{'Tag':<15} {'Recommended Tier':<20} {'Obj Pass':<10} {'Judge Pass':<12} {'Avg Score':<10} {'Cost (Base -> Rec)':<22} {'Savings':<15}"
        )
        print("-" * 80)
        for tag, rec in report.recommendations.items():
            cost_str = f"${rec.baseline_cost_usd:.4f} -> ${rec.recommended_cost_usd:.4f}"
            savings_str = f"{rec.estimated_savings_pct:.1f}% (${rec.estimated_savings_usd:.4f})"
            obj_str = (
                f"{rec.objective_pass_rate * 100:>7.1f}%"
                if rec.objective_pass_rate is not None
                else "    N/A"
            )
            judge_str = (
                f"{rec.judge_pass_rate * 100:>9.1f}%"
                if rec.judge_pass_rate is not None
                else "      N/A"
            )
            avg_str = f"{rec.avg_judge_score:.2f}" if rec.avg_judge_score is not None else "N/A"
            tier_display = f"{rec.recommended_tier}"
            print(
                f"{tag:<15} {tier_display:<20} {obj_str}  {judge_str}  {avg_str:<10} {cost_str:<22} {savings_str:<15}"
            )

        print("-" * 80)
        print("Candidate Tier Evaluation Details:")
        for tag, rec in report.recommendations.items():
            print(f"  [{tag}] Recommendation: {rec.recommended_tier} ({rec.decision_reason})")
            for ts in rec.tier_evaluations:
                status_str = "QUALIFIED" if ts.qualified else f"REJECTED ({ts.rejection_reason})"
                score_str = f"{ts.avg_judge_score:.2f}" if ts.avg_judge_score is not None else "N/A"
                print(
                    f"    - {ts.tier:<8}: Obj {ts.objective_pass_rate * 100:.1f}%, Judge {ts.judge_pass_rate * 100:.1f}%, Avg {score_str}, Cost ${ts.cost_usd:.4f} -> {status_str}"
                )

        print("=" * 80)
        print("Summary:")
        print(f"  Baseline Run Cost:              ${report.baseline_total_cost_usd:.4f}")
        print(f"  Recommended Routing Cost:       ${report.recommended_total_cost_usd:.4f}")
        print(
            f"  Estimated Replay Savings:       ${report.estimated_savings_usd:.4f} ({report.estimated_savings_pct:.1f}%)"
        )
        print(f"  {report.disclaimer}")
        print("=" * 80)
    finally:
        if close_session:
            session.close()


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""
    args_list = sys.argv[1:] if argv is None else argv

    if args_list and args_list[0] == "recommend":
        rec_parser = argparse.ArgumentParser(description="Downshift Model Router")
        rec_parser.add_argument("command", choices=["recommend"])
        rec_parser.add_argument("run_id", help="ID of the run to generate recommendations for")
        rec_parser.add_argument(
            "--min-pass-rate",
            type=float,
            default=0.95,
            help="Minimum semantic judge pass rate required (default: 0.95)",
        )
        rec_parser.add_argument(
            "--min-samples",
            type=int,
            default=3,
            help="Minimum evaluated samples required per tag (default: 3)",
        )
        rec_parser.add_argument(
            "--export",
            default=None,
            help="Path to export the tag-to-model routing map as JSON",
        )
        rec_parser.add_argument(
            "--json",
            action="store_true",
            help="Output full recommendation report as JSON",
        )
        args = rec_parser.parse_args(args_list)
        recommend_command(args)
        return

    if args_list and args_list[0] == "judge":
        judge_parser = argparse.ArgumentParser(description="Downshift Quality Judge")
        judge_parser.add_argument("command", choices=["judge"])
        judge_parser.add_argument("run_id", help="ID of the run to judge")
        judge_parser.add_argument("--model", default=None, help="Judge model identifier")
        judge_parser.add_argument(
            "--real", action="store_true", help="Execute against real Nebius Token Factory API"
        )
        judge_parser.add_argument(
            "--cost-ceiling",
            type=float,
            default=0.07,
            help="Hard aggregate cost ceiling in USD (default: 0.07)",
        )
        judge_parser.add_argument(
            "--force", action="store_true", help="Force re-judging existing evaluations"
        )
        args = judge_parser.parse_args(args_list)
        judge_command(args)
        return

    if args_list and args_list[0] == "evaluate":
        eval_parser = argparse.ArgumentParser(description="Downshift Objective Evaluator")
        eval_parser.add_argument("command", choices=["evaluate"])
        eval_parser.add_argument("run_id", help="ID of the run to evaluate")
        args = eval_parser.parse_args(args_list)
        evaluate_command(args.run_id)
        return

    if args_list and args_list[0] == "replay":
        replay_parser = argparse.ArgumentParser(description="Downshift Step Replayer")
        replay_parser.add_argument("command", choices=["replay"])
        replay_parser.add_argument("run_id", help="ID of the run to replay")
        replay_parser.add_argument(
            "--tiers",
            default="super,nano",
            help="Comma-separated tier list (default: super,nano)",
        )
        replay_parser.add_argument(
            "--real",
            action="store_true",
            help="Execute against real Nebius Token Factory API",
        )
        replay_parser.add_argument(
            "--max-workers",
            type=int,
            default=4,
            help="Maximum concurrent worker threads",
        )
        replay_parser.add_argument(
            "--max-steps",
            type=int,
            default=None,
            help="Maximum steps from the run to replay",
        )
        replay_parser.add_argument(
            "--cost-ceiling",
            type=float,
            default=None,
            help="Maximum USD cost allowed before aborting",
        )
        replay_parser.add_argument(
            "--force",
            action="store_true",
            help="Force replaying steps that were already replayed",
        )
        args = replay_parser.parse_args(args_list)
        replay_command(args)
        return

    parser = argparse.ArgumentParser(description="Downshift Run Inspector and Cost Verifier")
    parser.add_argument("run_id", nargs="?", help="ID of the run to inspect")
    parser.add_argument("--list", action="store_true", help="List all runs in the database")

    args = parser.parse_args(args_list)

    if args.list or not args.run_id:
        list_runs()
        if not args.run_id:
            return

    verified = inspect_run(args.run_id)
    if not verified:
        sys.exit(1)


if __name__ == "__main__":
    main()

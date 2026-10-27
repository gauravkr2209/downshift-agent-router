"""Tests for Phase 3 model routing recommendations, constraints, and CLI."""

import json
from pathlib import Path

import pytest
from sqlmodel import Session, select

from server.app import cli
from server.app import db as db_mod
from server.app.db import create_db_and_tables
from server.app.models import Evaluation, ModelPrice, Replay, Run, Step
from server.app.router import (
    _canonical_model_id,
    export_routing_config,
    recommend_routing,
    resolve_model_tier_name,
)


@pytest.fixture(autouse=True)
def seed_test_pricing() -> None:
    """Seed test pricing for Ultra, Super, and Nano models."""
    create_db_and_tables()
    with Session(db_mod.engine) as session:
        for model_id, in_p, out_p in [
            ("nvidia/Nemotron-3-Ultra-550b-a55b", 1.0, 3.0),
            ("nvidia/nemotron-3-super-120b-a12b", 0.30, 0.90),
            ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B", 0.06, 0.24),
            ("nemotron_ultra", 1.0, 3.0),
            ("nemotron_super", 0.30, 0.90),
            ("nemotron_nano", 0.06, 0.24),
        ]:
            existing = session.exec(select(ModelPrice).where(ModelPrice.model == model_id)).first()
            if not existing:
                session.add(ModelPrice(model=model_id, input_per_mtok=in_p, output_per_mtok=out_p))
        session.commit()


def _create_multi_step_run(
    session: Session,
    tag: str = "search",
    step_count: int = 3,
    baseline_step_cost: float = 0.003,
    baseline_model: str = "nvidia/Nemotron-3-Ultra-550b-a55b",
) -> tuple[str, list[str]]:
    """Helper to create a Run with multiple steps sharing a tag."""
    run = Run(
        task_name="test_routing_task",
        agent_name="test_agent",
        baseline_model=baseline_model,
        status="completed",
    )
    session.add(run)
    session.commit()
    session.refresh(run)

    step_ids: list[str] = []
    for i in range(step_count):
        step = Step(
            run_id=run.id,
            idx=i,
            tag=tag,
            kind="text",
            model=baseline_model,
            request={"messages": [{"role": "user", "content": f"Step {i}"}]},
            response={
                "choices": [{"finish_reason": "stop", "message": {"content": f"Step {i} ref"}}]
            },
            cost_usd=baseline_step_cost,
            tokens_in=100,
            tokens_out=50,
        )
        session.add(step)
        session.commit()
        session.refresh(step)
        step_ids.append(step.id)

    return run.id, step_ids


def _attach_replay_and_eval(
    session: Session,
    step_id: str,
    tier: str,
    model: str,
    cost_usd: float,
    objective_pass: bool = True,
    judge_pass: bool = True,
    judge_score: float | None = 0.95,
    judge_reason_prefix: str = "[PASS]",
    error: str | None = None,
    omit_evaluation: bool = False,
    empty_objective_detail: bool = False,
    omit_judge: bool = False,
) -> Replay:
    """Helper to attach a replay and evaluation to a step."""
    replay = Replay(
        step_id=step_id,
        tier=tier,
        model=model,
        mode="step_level",
        response={
            "choices": [{"finish_reason": "stop", "message": {"content": "Candidate output"}}]
        },
        cost_usd=cost_usd,
        tokens_in=100,
        tokens_out=50,
        error=error,
    )
    session.add(replay)
    session.commit()
    session.refresh(replay)

    if not omit_evaluation:
        detail = (
            {}
            if empty_objective_detail
            else {"checks": [{"name": "mock_check", "passed": objective_pass, "reason": "test"}]}
        )
        j_score = None if omit_judge else (judge_score if judge_pass else 0.4)
        j_reason = None if omit_judge else f"{judge_reason_prefix} Mock judgment"
        j_model = None if omit_judge else "meta-llama/llama-3.3-70b-instruct"
        ev = Evaluation(
            replay_id=replay.id,
            objective_pass=objective_pass,
            objective_detail=detail,
            judge_score=j_score,
            judge_reason=j_reason,
            judge_model=j_model,
        )
        session.add(ev)
        session.commit()
        session.refresh(ev)

    return replay


def test_cheapest_qualifying_tier_selected() -> None:
    """When both nano and super qualify, router selects nano as the cheapest."""
    with Session(db_mod.engine) as session:
        run_id, step_ids = _create_multi_step_run(
            session, tag="search", step_count=3, baseline_step_cost=0.003
        )

        for sid in step_ids:
            # Super: cost $0.001 per step, 100% pass
            _attach_replay_and_eval(
                session,
                sid,
                tier="super",
                model="nvidia/nemotron-3-super-120b-a12b",
                cost_usd=0.001,
            )
            # Nano: cost $0.0003 per step, 100% pass
            _attach_replay_and_eval(
                session,
                sid,
                tier="nano",
                model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
                cost_usd=0.0003,
            )

        report = recommend_routing(
            run_id=run_id, min_pass_rate=0.95, min_samples=3, session=session
        )

        assert "search" in report.recommendations
        rec = report.recommendations["search"]
        assert rec.recommended_tier == "nano"
        assert rec.recommended_model == "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
        assert rec.objective_pass_rate == 1.0
        assert rec.judge_pass_rate == 1.0
        assert rec.recommended_cost_usd == pytest.approx(0.0009)
        assert rec.baseline_cost_usd == pytest.approx(0.009)
        assert rec.estimated_savings_usd == pytest.approx(0.0081)
        assert rec.estimated_savings_pct == pytest.approx(90.0)


def test_objective_failure_hard_constraint() -> None:
    """Candidate with 100% judge score but failing objective check must be disqualified."""
    with Session(db_mod.engine) as session:
        run_id, step_ids = _create_multi_step_run(session, tag="search", step_count=3)

        for i, sid in enumerate(step_ids):
            # Step 0 fails objective checks, despite high judge score
            obj_ok = i != 0
            _attach_replay_and_eval(
                session,
                sid,
                tier="nano",
                model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
                cost_usd=0.0003,
                objective_pass=obj_ok,
                judge_pass=True,
                judge_score=1.0,
            )

        report = recommend_routing(
            run_id=run_id, min_pass_rate=0.95, min_samples=3, session=session
        )
        rec = report.recommendations["search"]

        # Nano disqualified due to objective check failure -> falls back to baseline
        assert rec.recommended_tier == "ultra"
        assert rec.recommended_model == "nvidia/Nemotron-3-Ultra-550b-a55b"
        assert rec.estimated_savings_usd == 0.0
        nano_summary = next(ts for ts in rec.tier_evaluations if ts.tier == "nano")
        assert not nano_summary.qualified
        assert "Objective failure" in (nano_summary.rejection_reason or "")


def test_judge_pass_rate_below_threshold() -> None:
    """Candidate meeting objective checks but falling below judge pass rate threshold is disqualified."""
    with Session(db_mod.engine) as session:
        run_id, step_ids = _create_multi_step_run(session, tag="synthesis", step_count=3)

        for i, sid in enumerate(step_ids):
            # 2 passes out of 3 = 66.7%, below 95% threshold
            j_pass = i != 2
            _attach_replay_and_eval(
                session,
                sid,
                tier="super",
                model="nvidia/nemotron-3-super-120b-a12b",
                cost_usd=0.001,
                objective_pass=True,
                judge_pass=j_pass,
                judge_reason_prefix="[PASS]" if j_pass else "[FAIL]",
            )

        report = recommend_routing(
            run_id=run_id, min_pass_rate=0.95, min_samples=3, session=session
        )
        rec = report.recommendations["synthesis"]

        assert rec.recommended_tier == "ultra"
        super_summary = next(ts for ts in rec.tier_evaluations if ts.tier == "super")
        assert not super_summary.qualified
        assert "Judge pass rate 66.7%" in (super_summary.rejection_reason or "")


def test_insufficient_sample_count() -> None:
    """When evaluated samples are fewer than min_samples (e.g. 2 < 3), baseline is retained."""
    with Session(db_mod.engine) as session:
        run_id, step_ids = _create_multi_step_run(session, tag="planning", step_count=2)

        for sid in step_ids:
            _attach_replay_and_eval(
                session,
                sid,
                tier="nano",
                model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
                cost_usd=0.0003,
                objective_pass=True,
                judge_pass=True,
            )

        # min_samples=3 but tag only has 2 steps
        report = recommend_routing(
            run_id=run_id, min_pass_rate=0.95, min_samples=3, session=session
        )
        rec = report.recommendations["planning"]

        assert rec.recommended_tier == "ultra"
        assert "Insufficient samples (2 < 3)" in rec.decision_reason


def test_uncertain_and_error_counted_as_failures() -> None:
    """Replays with [UNCERTAIN] or [ERROR] are treated as insufficient evidence, not passes."""
    with Session(db_mod.engine) as session:
        run_id, step_ids = _create_multi_step_run(session, tag="search", step_count=3)

        # 1 PASS, 1 UNCERTAIN, 1 ERROR
        prefixes = ["[PASS]", "[UNCERTAIN]", "[ERROR]"]
        for i, sid in enumerate(step_ids):
            _attach_replay_and_eval(
                session,
                sid,
                tier="nano",
                model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
                cost_usd=0.0003,
                objective_pass=True,
                judge_pass=(prefixes[i] == "[PASS]"),
                judge_reason_prefix=prefixes[i],
            )

        report = recommend_routing(
            run_id=run_id, min_pass_rate=0.95, min_samples=3, session=session
        )
        rec = report.recommendations["search"]

        assert rec.recommended_tier == "ultra"
        nano_summary = next(ts for ts in rec.tier_evaluations if ts.tier == "nano")
        assert not nano_summary.qualified
        # 1 / 3 passed = 33.3%
        assert nano_summary.judge_pass_rate == pytest.approx(1 / 3)


def test_fallback_to_baseline_when_no_cheaper_tier_qualifies() -> None:
    """Fallback to baseline with complete rejection explanations when all candidates fail."""
    with Session(db_mod.engine) as session:
        run_id, step_ids = _create_multi_step_run(session, tag="synthesis", step_count=3)

        for sid in step_ids:
            # Nano fails objective
            _attach_replay_and_eval(
                session,
                sid,
                tier="nano",
                model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
                cost_usd=0.0003,
                objective_pass=False,
            )
            # Super fails judge
            _attach_replay_and_eval(
                session,
                sid,
                tier="super",
                model="nvidia/nemotron-3-super-120b-a12b",
                cost_usd=0.001,
                objective_pass=True,
                judge_pass=False,
                judge_reason_prefix="[FAIL]",
            )

        report = recommend_routing(
            run_id=run_id, min_pass_rate=0.95, min_samples=3, session=session
        )
        rec = report.recommendations["synthesis"]

        assert rec.recommended_tier == "ultra"
        assert rec.estimated_savings_usd == 0.0
        assert "No cheaper tier qualified" in rec.decision_reason
        assert "nano: Objective failure" in rec.decision_reason
        assert "super: Judge pass rate" in rec.decision_reason


def test_missing_price_disqualifies_tier() -> None:
    """Candidate model with missing/null pricing cannot be recommended."""
    with Session(db_mod.engine) as session:
        run_id, step_ids = _create_multi_step_run(session, tag="search", step_count=3)

        # Insert price row with None values
        session.add(ModelPrice(model="unpriced-model", input_per_mtok=None, output_per_mtok=None))
        session.commit()

        for sid in step_ids:
            _attach_replay_and_eval(
                session,
                sid,
                tier="nano",
                model="unpriced-model",
                cost_usd=0.0003,
                objective_pass=True,
                judge_pass=True,
            )

        report = recommend_routing(
            run_id=run_id, min_pass_rate=0.95, min_samples=3, session=session
        )
        rec = report.recommendations["search"]

        assert rec.recommended_tier == "ultra"
        nano_summary = next(ts for ts in rec.tier_evaluations if ts.tier == "nano")
        assert not nano_summary.qualified
        assert "Pricing unconfigured" in (nano_summary.rejection_reason or "")


def test_incomplete_tier_coverage_disqualifies_tier() -> None:
    """Tier evaluated on only a subset of a tag's steps is rejected for incomplete coverage."""
    with Session(db_mod.engine) as session:
        run_id, step_ids = _create_multi_step_run(session, tag="search", step_count=3)

        # Replay only step 0 and 1, skipping step 2
        for sid in step_ids[:2]:
            _attach_replay_and_eval(
                session,
                sid,
                tier="nano",
                model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
                cost_usd=0.0003,
                objective_pass=True,
                judge_pass=True,
            )

        report = recommend_routing(
            run_id=run_id, min_pass_rate=0.95, min_samples=3, session=session
        )
        rec = report.recommendations["search"]

        assert rec.recommended_tier == "ultra"
        nano_summary = next(ts for ts in rec.tier_evaluations if ts.tier == "nano")
        assert not nano_summary.qualified
        assert "Incomplete tier coverage" in (nano_summary.rejection_reason or "")
        assert "1/3 steps" in (nano_summary.rejection_reason or "")


def test_matching_cost_comparison_no_unmatched_claim() -> None:
    """Cost comparison reflects matching coverage and does not claim unmatched savings."""
    with Session(db_mod.engine) as session:
        run_id, step_ids = _create_multi_step_run(
            session, tag="search", step_count=3, baseline_step_cost=0.005
        )

        for sid in step_ids:
            _attach_replay_and_eval(
                session,
                sid,
                tier="nano",
                model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
                cost_usd=0.001,
                objective_pass=True,
                judge_pass=True,
            )

        report = recommend_routing(
            run_id=run_id, min_pass_rate=0.95, min_samples=3, session=session
        )
        rec = report.recommendations["search"]

        # Baseline cost for 3 steps = 3 * 0.005 = 0.015
        # Nano cost for 3 steps = 3 * 0.001 = 0.003
        assert rec.baseline_cost_usd == pytest.approx(0.015)
        assert rec.recommended_cost_usd == pytest.approx(0.003)
        assert rec.estimated_savings_usd == pytest.approx(0.012)
        assert rec.estimated_savings_pct == pytest.approx(80.0)


def test_zero_cost_safeguard() -> None:
    """Zero baseline cost computes 0.0% savings cleanly without ZeroDivisionError."""
    with Session(db_mod.engine) as session:
        run_id, step_ids = _create_multi_step_run(
            session, tag="search", step_count=3, baseline_step_cost=0.0
        )

        for sid in step_ids:
            _attach_replay_and_eval(
                session,
                sid,
                tier="nano",
                model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
                cost_usd=0.0,
                objective_pass=True,
                judge_pass=True,
            )

        report = recommend_routing(
            run_id=run_id, min_pass_rate=0.95, min_samples=3, session=session
        )
        rec = report.recommendations["search"]

        assert rec.baseline_cost_usd == 0.0
        assert rec.estimated_savings_pct == 0.0


def test_export_flag_writes_valid_json(tmp_path: Path) -> None:
    """Exporting routing report produces a valid JSON map file."""
    with Session(db_mod.engine) as session:
        run_id, step_ids = _create_multi_step_run(session, tag="search", step_count=3)

        for sid in step_ids:
            _attach_replay_and_eval(
                session,
                sid,
                tier="nano",
                model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
                cost_usd=0.0003,
                objective_pass=True,
                judge_pass=True,
            )

        report = recommend_routing(
            run_id=run_id, min_pass_rate=0.95, min_samples=3, session=session
        )
        export_path = tmp_path / "routing_map.json"
        export_routing_config(report, export_path)

        assert export_path.exists()
        with open(export_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data == {"search": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"}


def test_cli_recommend_command(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI recommend command executes and outputs formatted table and export."""
    with Session(db_mod.engine) as session:
        run_id, step_ids = _create_multi_step_run(session, tag="search", step_count=3)

        for sid in step_ids:
            _attach_replay_and_eval(
                session,
                sid,
                tier="nano",
                model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
                cost_usd=0.0003,
                objective_pass=True,
                judge_pass=True,
            )

    export_file = tmp_path / "cli_export.json"

    # Test unknown run ID failure
    with pytest.raises(SystemExit) as exc:
        cli.main(["recommend", "nonexistent-run-id"])
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "Error: Run 'nonexistent-run-id' not found." in err

    # Test valid run with export and table output
    cli.main(["recommend", run_id, "--export", str(export_file)])
    out = capsys.readouterr().out

    assert f"Downshift Model Routing Recommendations: Run {run_id}" in out
    assert "Threshold: 95.0% pass rate | Min samples: 3" in out
    assert "search" in out
    assert "nano" in out
    assert "Estimated Replay Savings:" in out
    assert "Disclaimer: These savings are estimated from isolated step-level replays" in out
    assert export_file.exists()


def test_exact_replay_coverage_duplicate_replays_disqualifies() -> None:
    """Duplicate replays for a step prevent tier qualification even if total count matches."""
    with Session(db_mod.engine) as session:
        run_id, step_ids = _create_multi_step_run(session, tag="search", step_count=3)

        # 2 replays for step_ids[0], 1 for step_ids[1], 0 for step_ids[2]
        # Total replays = 3, but duplicates exist and step_ids[2] is missing!
        _attach_replay_and_eval(
            session,
            step_ids[0],
            tier="nano",
            model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
            cost_usd=0.0003,
            objective_pass=True,
            judge_pass=True,
        )
        _attach_replay_and_eval(
            session,
            step_ids[0],
            tier="nano",
            model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
            cost_usd=0.0003,
            objective_pass=True,
            judge_pass=True,
        )
        _attach_replay_and_eval(
            session,
            step_ids[1],
            tier="nano",
            model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
            cost_usd=0.0003,
            objective_pass=True,
            judge_pass=True,
        )

        report = recommend_routing(
            run_id=run_id, min_pass_rate=0.95, min_samples=3, session=session
        )
        rec = report.recommendations["search"]

        assert rec.recommended_tier == "ultra"
        nano_summary = next(ts for ts in rec.tier_evaluations if ts.tier == "nano")
        assert not nano_summary.qualified
        assert "Incomplete tier coverage" in (
            nano_summary.rejection_reason or ""
        ) or "Duplicate replays" in (nano_summary.rejection_reason or "")


def test_exact_replay_coverage_replay_error_disqualifies() -> None:
    """A replay error on any step prevents tier qualification."""
    with Session(db_mod.engine) as session:
        run_id, step_ids = _create_multi_step_run(session, tag="search", step_count=3)

        for i, sid in enumerate(step_ids):
            err = "Simulated 500 error" if i == 1 else None
            _attach_replay_and_eval(
                session,
                sid,
                tier="nano",
                model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
                cost_usd=0.0003,
                objective_pass=(err is None),
                judge_pass=(err is None),
                error=err,
            )

        report = recommend_routing(
            run_id=run_id, min_pass_rate=0.95, min_samples=3, session=session
        )
        rec = report.recommendations["search"]

        assert rec.recommended_tier == "ultra"
        nano_summary = next(ts for ts in rec.tier_evaluations if ts.tier == "nano")
        assert not nano_summary.qualified
        assert "Replay error" in (nano_summary.rejection_reason or "")


def test_cli_recommend_json_with_export_stdout_is_pure_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When --json and --export are combined, stdout must be strictly valid parseable JSON."""
    with Session(db_mod.engine) as session:
        run_id, step_ids = _create_multi_step_run(session, tag="search", step_count=3)

        for sid in step_ids:
            _attach_replay_and_eval(
                session,
                sid,
                tier="nano",
                model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
                cost_usd=0.0003,
                objective_pass=True,
                judge_pass=True,
            )

    export_file = tmp_path / "routing_export.json"
    cli.main(["recommend", run_id, "--json", "--export", str(export_file)])

    captured = capsys.readouterr()
    # stdout must parse cleanly as JSON without JSONDecodeError!
    parsed_json = json.loads(captured.out)
    assert isinstance(parsed_json, dict)
    assert parsed_json["run_id"] == run_id
    assert "recommendations" in parsed_json

    # export confirmation must be in stderr, leaving stdout clean
    assert "Exported routing map to:" in captured.err
    assert export_file.exists()


def test_input_validation_cli_and_function(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Invalid min-pass-rate and min-samples are rejected with clear errors."""
    with Session(db_mod.engine) as session:
        run_id, _ = _create_multi_step_run(session, tag="search", step_count=3)

    # Function-level validation
    with pytest.raises(ValueError, match="--min-pass-rate must be between 0.0 and 1.0"):
        recommend_routing(run_id=run_id, min_pass_rate=1.5)

    with pytest.raises(ValueError, match="--min-pass-rate must be between 0.0 and 1.0"):
        recommend_routing(run_id=run_id, min_pass_rate=-0.1)

    with pytest.raises(ValueError, match="--min-samples must be at least 1"):
        recommend_routing(run_id=run_id, min_samples=0)

    # CLI-level validation
    with pytest.raises(SystemExit) as exc:
        cli.main(["recommend", run_id, "--min-pass-rate", "1.5"])
    assert exc.value.code == 1
    assert "Error: --min-pass-rate must be between 0.0 and 1.0" in capsys.readouterr().err

    with pytest.raises(SystemExit) as exc:
        cli.main(["recommend", run_id, "--min-samples", "0"])
    assert exc.value.code == 1
    assert "Error: --min-samples must be at least 1" in capsys.readouterr().err


def test_baseline_metrics_honestly_represented_as_none(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When baseline is retained, unmeasured baseline metrics are honestly None and displayed as N/A."""
    with Session(db_mod.engine) as session:
        run_id, _ = _create_multi_step_run(session, tag="planning", step_count=3)

        # No candidate replays attached; baseline is retained
        report = recommend_routing(
            run_id=run_id, min_pass_rate=0.95, min_samples=3, session=session
        )
        rec = report.recommendations["planning"]

        assert rec.recommended_tier == "ultra"
        # Baseline metrics were not measured by replays, so must be None!
        assert rec.objective_pass_rate is None
        assert rec.judge_pass_rate is None
        assert rec.avg_judge_score is None

    # Verify CLI table formats None metrics as N/A
    cli.main(["recommend", run_id])
    out = capsys.readouterr().out
    assert "planning" in out
    assert "ultra" in out
    assert "N/A" in out


def test_replay_for_wrong_step_cannot_count_toward_coverage() -> None:
    """A replay for a step outside the target tag cannot satisfy coverage for the tag."""
    with Session(db_mod.engine) as session:
        run_id, search_step_ids = _create_multi_step_run(session, tag="search", step_count=3)
        # Create a separate step under tag 'synthesis'
        run = session.get(Run, run_id)
        assert run is not None
        synth_step = Step(
            run_id=run_id,
            idx=3,
            tag="synthesis",
            kind="text",
            model="nvidia/Nemotron-3-Ultra-550b-a55b",
            request={"messages": [{"role": "user", "content": "Synth step"}]},
            response={"choices": [{"finish_reason": "stop", "message": {"content": "Synth ref"}}]},
            cost_usd=0.003,
            tokens_in=100,
            tokens_out=50,
        )
        session.add(synth_step)
        session.commit()
        session.refresh(synth_step)

        # Replay search steps 0 and 1, plus synthesis step 3 (wrong step for search)
        _attach_replay_and_eval(
            session,
            search_step_ids[0],
            tier="nano",
            model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
            cost_usd=0.0003,
            objective_pass=True,
            judge_pass=True,
        )
        _attach_replay_and_eval(
            session,
            search_step_ids[1],
            tier="nano",
            model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
            cost_usd=0.0003,
            objective_pass=True,
            judge_pass=True,
        )
        _attach_replay_and_eval(
            session,
            synth_step.id,
            tier="nano",
            model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
            cost_usd=0.0003,
            objective_pass=True,
            judge_pass=True,
        )

        report = recommend_routing(
            run_id=run_id, min_pass_rate=0.95, min_samples=3, session=session
        )
        search_rec = report.recommendations["search"]

        # Nano must NOT qualify for 'search' because search_step_ids[2] is missing!
        assert search_rec.recommended_tier == "ultra"
        nano_summary = next(ts for ts in search_rec.tier_evaluations if ts.tier == "nano")
        assert not nano_summary.qualified
        assert "Incomplete tier coverage" in (nano_summary.rejection_reason or "")


def test_candidate_cannot_qualify_with_missing_replay_when_baseline_has_enough_steps() -> None:
    """Candidate cannot qualify when a replay is missing, even if baseline has >= min_samples steps."""
    with Session(db_mod.engine) as session:
        # Baseline has 4 steps, min_samples is 3
        run_id, step_ids = _create_multi_step_run(session, tag="search", step_count=4)

        # Candidate tier only replays 3 steps (step_ids[3] missing)
        for sid in step_ids[:3]:
            _attach_replay_and_eval(
                session,
                sid,
                tier="nano",
                model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
                cost_usd=0.0003,
                objective_pass=True,
                judge_pass=True,
            )

        report = recommend_routing(
            run_id=run_id, min_pass_rate=0.95, min_samples=3, session=session
        )
        rec = report.recommendations["search"]

        assert rec.recommended_tier == "ultra"
        nano_summary = next(ts for ts in rec.tier_evaluations if ts.tier == "nano")
        assert not nano_summary.qualified
        assert "Incomplete tier coverage" in (nano_summary.rejection_reason or "")
        assert "missing replay for 1/4 steps" in (nano_summary.rejection_reason or "")
        assert "Incomplete tier coverage" in rec.decision_reason


def test_candidate_cannot_qualify_with_unevaluated_replay_when_baseline_has_enough_steps() -> None:
    """Candidate cannot qualify when a replay lacks an evaluation, even if baseline has >= min_samples."""
    with Session(db_mod.engine) as session:
        # Baseline has 4 steps, min_samples is 3
        run_id, step_ids = _create_multi_step_run(session, tag="search", step_count=4)

        # 4 replays exist, but step_ids[3] has omit_evaluation=True
        for i, sid in enumerate(step_ids):
            _attach_replay_and_eval(
                session,
                sid,
                tier="nano",
                model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
                cost_usd=0.0003,
                objective_pass=True,
                judge_pass=True,
                omit_evaluation=(i == 3),
            )

        report = recommend_routing(
            run_id=run_id, min_pass_rate=0.95, min_samples=3, session=session
        )
        rec = report.recommendations["search"]

        assert rec.recommended_tier == "ultra"
        nano_summary = next(ts for ts in rec.tier_evaluations if ts.tier == "nano")
        assert not nano_summary.qualified
        assert "Missing or incomplete evaluation" in (nano_summary.rejection_reason or "")
        assert "1/4 replay(s)" in (nano_summary.rejection_reason or "")
        assert nano_summary.samples == 3
        assert nano_summary.total_steps == 4
        assert "Missing or incomplete evaluation" in rec.decision_reason


def test_candidate_cannot_qualify_when_objective_checks_not_run() -> None:
    """Candidate cannot qualify when objective checks have not actually executed."""
    with Session(db_mod.engine) as session:
        run_id, step_ids = _create_multi_step_run(session, tag="search", step_count=4)

        for i, sid in enumerate(step_ids):
            _attach_replay_and_eval(
                session,
                sid,
                tier="nano",
                model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
                cost_usd=0.0003,
                objective_pass=True,
                judge_pass=True,
                empty_objective_detail=(i == 2),  # Objective checks never ran for step 2
            )

        report = recommend_routing(
            run_id=run_id, min_pass_rate=0.95, min_samples=3, session=session
        )
        rec = report.recommendations["search"]

        assert rec.recommended_tier == "ultra"
        nano_summary = next(ts for ts in rec.tier_evaluations if ts.tier == "nano")
        assert not nano_summary.qualified
        assert "Missing or incomplete evaluation" in (nano_summary.rejection_reason or "")
        assert "1/4 replay(s)" in (nano_summary.rejection_reason or "")


def test_candidate_cannot_qualify_when_judge_not_run() -> None:
    """Candidate cannot qualify when judge checks have not executed on all replays."""
    with Session(db_mod.engine) as session:
        run_id, step_ids = _create_multi_step_run(session, tag="search", step_count=4)

        for i, sid in enumerate(step_ids):
            _attach_replay_and_eval(
                session,
                sid,
                tier="nano",
                model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
                cost_usd=0.0003,
                objective_pass=True,
                judge_pass=True,
                omit_judge=(i == 1),  # Judge never evaluated step 1
            )

        report = recommend_routing(
            run_id=run_id, min_pass_rate=0.95, min_samples=3, session=session
        )
        rec = report.recommendations["search"]

        assert rec.recommended_tier == "ultra"
        nano_summary = next(ts for ts in rec.tier_evaluations if ts.tier == "nano")
        assert not nano_summary.qualified
        assert "Missing or incomplete evaluation" in (nano_summary.rejection_reason or "")
        assert "1/4 replay(s)" in (nano_summary.rejection_reason or "")


def test_baseline_tier_dynamically_resolved_from_model() -> None:
    """Baseline tier is resolved dynamically from model name rather than hardcoded to ultra."""
    with Session(db_mod.engine) as session:
        # 1. Baseline is Nemotron Super
        run_id_super, _ = _create_multi_step_run(
            session,
            tag="planning",
            step_count=3,
            baseline_model="nvidia/nemotron-3-super-120b-a12b",
        )
        report_super = recommend_routing(run_id=run_id_super, session=session)
        rec_super = report_super.recommendations["planning"]
        assert rec_super.baseline_tier == "super"
        assert rec_super.baseline_model == "nvidia/nemotron-3-super-120b-a12b"
        # When no candidate qualifies, retained tier matches baseline tier 'super'
        assert rec_super.recommended_tier == "super"

        # 2. Baseline is Nemotron Nano
        run_id_nano, _ = _create_multi_step_run(
            session,
            tag="planning",
            step_count=3,
            baseline_model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
        )
        report_nano = recommend_routing(run_id=run_id_nano, session=session)
        rec_nano = report_nano.recommendations["planning"]
        assert rec_nano.baseline_tier == "nano"
        assert rec_nano.baseline_model == "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
        assert rec_nano.recommended_tier == "nano"

        # 3. Baseline is a custom model
        run_id_custom, _ = _create_multi_step_run(
            session,
            tag="planning",
            step_count=3,
            baseline_model="meta-llama/Llama-3-70b-instruct",
        )
        report_custom = recommend_routing(run_id=run_id_custom, session=session)
        rec_custom = report_custom.recommendations["planning"]
        assert rec_custom.baseline_tier == "meta-llama/Llama-3-70b-instruct"
        assert rec_custom.recommended_tier == "meta-llama/Llama-3-70b-instruct"

    # Direct helper verification
    assert resolve_model_tier_name("nvidia/Nemotron-3-Ultra-550b-a55b") == "ultra"
    assert resolve_model_tier_name("nemotron_ultra") == "ultra"
    assert resolve_model_tier_name("nvidia/nemotron-3-super-120b-a12b") == "super"
    assert resolve_model_tier_name("nemotron_super") == "super"
    assert resolve_model_tier_name("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B") == "nano"
    assert resolve_model_tier_name("nemotron_nano") == "nano"
    assert resolve_model_tier_name("custom-provider/custom-model") == "custom-provider/custom-model"


def test_baseline_model_excluded_from_downgrade_candidates() -> None:
    """Baseline model appearing in candidate replays cannot qualify as a downgrade."""
    with Session(db_mod.engine) as session:
        # Baseline model is Nemotron Ultra, cost $0.003/step
        run_id, step_ids = _create_multi_step_run(
            session,
            tag="search",
            step_count=3,
            baseline_step_cost=0.003,
            baseline_model="nvidia/Nemotron-3-Ultra-550b-a55b",
        )

        # 1. Attach replay for tier 'ultra' using the baseline model directly
        # Set replay cost lower than baseline ($0.001 vs $0.003) to prove cost alone doesn't qualify it
        for sid in step_ids:
            _attach_replay_and_eval(
                session,
                sid,
                tier="ultra",
                model="nvidia/Nemotron-3-Ultra-550b-a55b",
                cost_usd=0.001,
                objective_pass=True,
                judge_pass=True,
            )

        # 2. Attach another replay with a custom tier name but baseline alias model 'nemotron_ultra'
        for sid in step_ids:
            _attach_replay_and_eval(
                session,
                sid,
                tier="candidate_baseline_rerun",
                model="nemotron_ultra",
                cost_usd=0.001,
                objective_pass=True,
                judge_pass=True,
            )

        report = recommend_routing(
            run_id=run_id, min_pass_rate=0.95, min_samples=3, session=session
        )
        rec = report.recommendations["search"]

        # Neither baseline candidate qualifies as a downgrade
        for ts in rec.tier_evaluations:
            assert not ts.qualified
            assert (
                ts.rejection_reason == "Baseline model cannot be recommended as its own downgrade"
            )

        # Router falls back to baseline, not claiming false savings against itself
        assert rec.recommended_tier == "ultra"
        assert rec.recommended_model == "nvidia/Nemotron-3-Ultra-550b-a55b"
        assert rec.estimated_savings_usd == 0.0
        assert rec.estimated_savings_pct == 0.0
        assert "Baseline model cannot be recommended as its own downgrade" in rec.decision_reason
        assert report.routing_map["search"] == "nvidia/Nemotron-3-Ultra-550b-a55b"


def test_routing_map_consistency_multi_tag_fallback_and_selection(tmp_path: Path) -> None:
    """Routing map consistency: winning candidates map to real model IDs and failing tags fall back."""
    with Session(db_mod.engine) as session:
        # Create a run with two distinct tags: 'search' and 'synthesis'
        run = Run(
            task_name="test_multi_tag_task",
            agent_name="test_agent",
            baseline_model="nvidia/Nemotron-3-Ultra-550b-a55b",
            status="completed",
        )
        session.add(run)
        session.commit()
        session.refresh(run)

        # Tag 1: search (3 steps)
        search_step_ids: list[str] = []
        for i in range(3):
            s = Step(
                run_id=run.id,
                idx=i,
                tag="search",
                kind="text",
                model="nvidia/Nemotron-3-Ultra-550b-a55b",
                request={"messages": [{"role": "user", "content": f"Search step {i}"}]},
                response={"choices": [{"finish_reason": "stop", "message": {"content": "ref"}}]},
                cost_usd=0.003,
                tokens_in=100,
                tokens_out=50,
            )
            session.add(s)
            session.commit()
            session.refresh(s)
            search_step_ids.append(s.id)

        # Tag 2: synthesis (3 steps)
        synth_step_ids: list[str] = []
        for i in range(3, 6):
            s = Step(
                run_id=run.id,
                idx=i,
                tag="synthesis",
                kind="text",
                model="nvidia/Nemotron-3-Ultra-550b-a55b",
                request={"messages": [{"role": "user", "content": f"Synth step {i}"}]},
                response={"choices": [{"finish_reason": "stop", "message": {"content": "ref"}}]},
                cost_usd=0.003,
                tokens_in=100,
                tokens_out=50,
            )
            session.add(s)
            session.commit()
            session.refresh(s)
            synth_step_ids.append(s.id)

        # 'search' tag: nano qualifies
        for sid in search_step_ids:
            _attach_replay_and_eval(
                session,
                sid,
                tier="nano",
                model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
                cost_usd=0.0003,
                objective_pass=True,
                judge_pass=True,
            )

        # 'synthesis' tag: nano fails objective check -> no candidate qualifies
        for sid in synth_step_ids:
            _attach_replay_and_eval(
                session,
                sid,
                tier="nano",
                model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
                cost_usd=0.0003,
                objective_pass=False,
                judge_pass=True,
            )

        report = recommend_routing(
            run_id=run.id, min_pass_rate=0.95, min_samples=3, session=session
        )

        # Routing map checks
        assert report.routing_map["search"] == "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
        assert report.routing_map["synthesis"] == "nvidia/Nemotron-3-Ultra-550b-a55b"

        # Export and verify JSON consistency
        export_file = tmp_path / "multi_tag_routing_map.json"
        export_routing_config(report, export_file)
        assert export_file.exists()

        with open(export_file, "r", encoding="utf-8") as f:
            exported_data = json.load(f)

        assert exported_data == {
            "search": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
            "synthesis": "nvidia/Nemotron-3-Ultra-550b-a55b",
        }


def test_baseline_exclusion_works_both_ways_alias_and_canonical() -> None:
    """Baseline exclusion works when baseline is alias and candidate is canonical, and vice versa."""
    with Session(db_mod.engine) as session:
        # Case 1: Baseline is specified by alias 'nemotron_ultra', candidate uses canonical ID
        run_id_alias_base, step_ids_1 = _create_multi_step_run(
            session,
            tag="planning",
            step_count=3,
            baseline_step_cost=0.003,
            baseline_model="nemotron_ultra",
        )
        for sid in step_ids_1:
            _attach_replay_and_eval(
                session,
                sid,
                tier="candidate_eval",
                model="nvidia/Nemotron-3-Ultra-550b-a55b",
                cost_usd=0.001,
                objective_pass=True,
                judge_pass=True,
            )

        report_1 = recommend_routing(run_id=run_id_alias_base, session=session)
        rec_1 = report_1.recommendations["planning"]
        assert not rec_1.tier_evaluations[0].qualified
        assert (
            rec_1.tier_evaluations[0].rejection_reason
            == "Baseline model cannot be recommended as its own downgrade"
        )
        assert rec_1.recommended_tier == "ultra"
        assert rec_1.recommended_model == "nvidia/Nemotron-3-Ultra-550b-a55b"
        assert report_1.routing_map["planning"] == "nvidia/Nemotron-3-Ultra-550b-a55b"

        # Case 2: Baseline is specified by short alias 'ultra', candidate uses canonical ID
        run_id_short_alias, step_ids_2 = _create_multi_step_run(
            session,
            tag="analysis",
            step_count=3,
            baseline_step_cost=0.003,
            baseline_model="ultra",
        )
        for sid in step_ids_2:
            _attach_replay_and_eval(
                session,
                sid,
                tier="candidate_eval",
                model="nvidia/Nemotron-3-Ultra-550b-a55b",
                cost_usd=0.001,
                objective_pass=True,
                judge_pass=True,
            )

        report_2 = recommend_routing(run_id=run_id_short_alias, session=session)
        rec_2 = report_2.recommendations["analysis"]
        assert not rec_2.tier_evaluations[0].qualified
        assert (
            rec_2.tier_evaluations[0].rejection_reason
            == "Baseline model cannot be recommended as its own downgrade"
        )
        assert report_2.routing_map["analysis"] == "nvidia/Nemotron-3-Ultra-550b-a55b"

        # Direct unit test of _canonical_model_id edge cases
        assert _canonical_model_id("  nvidia/Nemotron-3-Ultra-550b-a55b  ", session) == (
            "nvidia/Nemotron-3-Ultra-550b-a55b"
        )
        assert _canonical_model_id("  nemotron_ultra  ", session) == (
            "nvidia/Nemotron-3-Ultra-550b-a55b"
        )
        assert _canonical_model_id("  ultra  ", session) == ("nvidia/Nemotron-3-Ultra-550b-a55b")
        assert _canonical_model_id("  NEMOTRON_SUPER  ", session) == (
            "nvidia/nemotron-3-super-120b-a12b"
        )
        assert _canonical_model_id("  super  ", session) == ("nvidia/nemotron-3-super-120b-a12b")
        assert _canonical_model_id("  nemotron_nano  ", session) == (
            "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
        )
        assert _canonical_model_id("  nano  ", session) == ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B")
        assert _canonical_model_id("  custom-org/unknown-model  ", session) == (
            "custom-org/unknown-model"
        )
        assert _canonical_model_id("  ", session) == ""
        assert _canonical_model_id("", session) == ""


def test_unpriced_baseline_and_unknown_candidate_handling() -> None:
    """When baseline and candidate share unpriced model ID, candidate is excluded as downgrade.

    Also verifies unknown candidate model ID remains unchanged and cannot qualify without pricing.
    """
    with Session(db_mod.engine) as session:
        # Part 1: Baseline and candidate share unpriced custom model ID
        unpriced_model = "custom-org/unpriced-baseline-model"
        run_id_unpriced, step_ids = _create_multi_step_run(
            session,
            tag="planning",
            step_count=3,
            baseline_step_cost=0.003,
            baseline_model=unpriced_model,
        )

        for sid in step_ids:
            _attach_replay_and_eval(
                session,
                sid,
                tier="custom_tier",
                model=unpriced_model,
                cost_usd=0.001,
                objective_pass=True,
                judge_pass=True,
            )

        report = recommend_routing(run_id=run_id_unpriced, session=session)
        rec = report.recommendations["planning"]

        # Even with failed/unavailable pricing lookup, candidate is recognized as baseline
        summary = rec.tier_evaluations[0]
        assert not summary.qualified
        assert (
            summary.rejection_reason == "Baseline model cannot be recommended as its own downgrade"
        )
        assert rec.recommended_tier == unpriced_model
        assert rec.recommended_model == unpriced_model
        assert rec.estimated_savings_usd == 0.0
        assert report.routing_map["planning"] == unpriced_model

        # Part 2: Unknown candidate model ID remains unchanged and is rejected without pricing
        unknown_model = "custom-org/completely-unknown-candidate"
        assert _canonical_model_id(unknown_model, session) == unknown_model

        run_id_ultra, ultra_step_ids = _create_multi_step_run(
            session,
            tag="search",
            step_count=3,
            baseline_step_cost=0.003,
            baseline_model="nvidia/Nemotron-3-Ultra-550b-a55b",
        )

        for sid in ultra_step_ids:
            _attach_replay_and_eval(
                session,
                sid,
                tier="unknown_tier",
                model=unknown_model,
                cost_usd=0.0005,
                objective_pass=True,
                judge_pass=True,
            )

        report_ultra = recommend_routing(run_id=run_id_ultra, session=session)
        rec_ultra = report_ultra.recommendations["search"]
        unknown_summary = rec_ultra.tier_evaluations[0]

        # Model ID is preserved without being converted into a misleading identity
        assert unknown_summary.model == unknown_model
        assert not unknown_summary.qualified
        assert (
            unknown_summary.rejection_reason == f"Pricing unconfigured for model '{unknown_model}'"
        )
        assert rec_ultra.recommended_tier == "ultra"
        assert rec_ultra.recommended_model == "nvidia/Nemotron-3-Ultra-550b-a55b"
        assert report_ultra.routing_map["search"] == "nvidia/Nemotron-3-Ultra-550b-a55b"

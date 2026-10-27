"""Tests for token usage and estimated cost database audit."""

from collections.abc import Generator
from pathlib import Path

import pytest
from sqlmodel import Session, SQLModel, create_engine

from scripts.audit_token_usage import run_token_audit
from server.app.models import Replay, Run, Step


@pytest.fixture
def test_db_session() -> Generator[Session, None, None]:
    """Provide an isolated in-memory database session."""
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def test_audit_empty_database(test_db_session: Session, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify audit handles an empty database safely and discloses unavailable fields."""
    metrics = run_token_audit(session=test_db_session)
    assert metrics["runs_count"] == 0
    assert metrics["steps_count"] == 0
    assert metrics["replays_count"] == 0
    assert metrics["step_total_tokens"] == 0
    assert metrics["step_total_cost_usd"] == 0.0

    captured = capsys.readouterr()
    assert "Database is currently empty" in captured.out
    assert "UNAVAILABLE in DB" in captured.out


def test_audit_aggregations_and_anti_double_counting(
    test_db_session: Session, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify steps and replays are computed accurately and never double-counted into run totals."""
    # Create Run 1
    run1 = Run(
        id="run-test-001",
        task_name="test_task_1",
        agent_name="test_agent",
        baseline_model="nvidia/nemotron-3-ultra-550b-a55b",
        status="completed",
        tokens_in=300,
        tokens_out=130,
        total_cost_usd=0.00028,
    )
    test_db_session.add(run1)

    # Step 0: Ultra tier
    s0 = Step(
        id="step-test-001",
        run_id="run-test-001",
        idx=0,
        tag="plan",
        kind="text",
        model="nvidia/nemotron-3-ultra-550b-a55b",
        tokens_in=100,
        tokens_out=50,
        cost_usd=0.00025,
        response={"usage": {"prompt_tokens": 100, "completion_tokens": 50}},
    )
    # Step 1: Nano tier
    s1 = Step(
        id="step-test-002",
        run_id="run-test-001",
        idx=1,
        tag="summarize_page",
        kind="tool_call",
        model="nvidia/nemotron-3-nano-30b-a3b",
        tokens_in=200,
        tokens_out=80,
        cost_usd=0.00003,
        response={"usage": {"prompt_tokens": 200, "completion_tokens": 80}},
    )
    test_db_session.add(s0)
    test_db_session.add(s1)

    # Replay on Step 0 using Super tier (candidate evaluation call)
    r0 = Replay(
        id="replay-test-001",
        step_id="step-test-001",
        tier="super",
        model="nvidia/nemotron-3-super-120b-a12b",
        tokens_in=100,
        tokens_out=40,
        cost_usd=0.00006,
        response={"usage": {"prompt_tokens": 100, "completion_tokens": 40}},
    )
    test_db_session.add(r0)
    test_db_session.commit()

    metrics = run_token_audit(session=test_db_session)

    # Verify primary step totals
    assert metrics["runs_count"] == 1
    assert metrics["steps_count"] == 2
    assert metrics["step_tokens_in"] == 300
    assert metrics["step_tokens_out"] == 130
    assert metrics["step_total_tokens"] == 430
    assert metrics["step_total_cost_usd"] == pytest.approx(0.00028)
    assert metrics["missing_usage_steps"] == 0

    # Verify replays are tracked separately and NOT double-counted into step totals
    assert metrics["replays_count"] == 1
    assert metrics["replay_total_tokens"] == 140
    assert metrics["replay_total_cost_usd"] == pytest.approx(0.00006)

    captured = capsys.readouterr()
    assert "ULTRA" in captured.out
    assert "NANO" in captured.out
    assert "plan" in captured.out
    assert "summarize_page" in captured.out
    assert "Replay calls are tracked separately and NOT double-counted" in captured.out


def test_audit_missing_usage_marked_unknown_not_zero(
    test_db_session: Session, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify that steps with missing usage metadata are flagged as missing and not counted as zero."""
    run_err = Run(
        id="run-test-err",
        task_name="failed_task",
        agent_name="test_agent",
        baseline_model="nvidia/nemotron-3-ultra-550b-a55b",
        status="failed",
        tokens_in=100,
        tokens_out=50,
        total_cost_usd=0.00025,
        meta={"has_unreconciled_cost": True, "unreconciled_reason": "Missing usage"},
    )
    test_db_session.add(run_err)

    # Valid step
    s_ok = Step(
        id="step-ok",
        run_id="run-test-err",
        idx=0,
        tag="plan",
        kind="text",
        model="nvidia/nemotron-3-ultra-550b-a55b",
        tokens_in=100,
        tokens_out=50,
        cost_usd=0.00025,
        response={"usage": {"prompt_tokens": 100, "completion_tokens": 50}},
    )
    # Step where usage was absent from provider response
    s_missing = Step(
        id="step-missing",
        run_id="run-test-err",
        idx=1,
        tag="search_query",
        kind="error",
        model="nvidia/nemotron-3-ultra-550b-a55b",
        tokens_in=0,
        tokens_out=0,
        cost_usd=0.0,
        response={"usage": None},
    )
    test_db_session.add(s_ok)
    test_db_session.add(s_missing)
    test_db_session.commit()

    metrics = run_token_audit(session=test_db_session)

    # Step tokens should only sum the reconciled step (100), NOT count missing as verified 0
    assert metrics["steps_count"] == 2
    assert metrics["missing_usage_steps"] == 1
    assert metrics["step_tokens_in"] == 100
    assert metrics["step_tokens_out"] == 50

    captured = capsys.readouterr()
    assert "Missing-Usage Steps     : 1" in captured.out
    assert "failed (UNREC)" in captured.out


def test_audit_timeout_error_step_flagged_as_unknown(
    test_db_session: Session, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify that a step recorded during an APITimeoutError is audited as missing usage / UNKNOWN."""
    run_timeout = Run(
        id="run-timeout-audit",
        task_name="timeout_audit_task",
        agent_name="test_agent",
        baseline_model="nvidia/nemotron-3-ultra-550b-a55b",
        status="failed",
        tokens_in=0,
        tokens_out=0,
        total_cost_usd=0.0,
        meta={
            "has_unreconciled_cost": True,
            "unreconciled_reason": "Call timed out (APITimeoutError: Request timed out).",
        },
    )
    test_db_session.add(run_timeout)

    s_timeout = Step(
        id="step-timeout-01",
        run_id="run-timeout-audit",
        idx=0,
        tag="ultra_tool_call",
        kind="error",
        model="nvidia/nemotron-3-ultra-550b-a55b",
        tokens_in=0,
        tokens_out=0,
        cost_usd=0.0,
        response={"error": "Request timed out", "error_type": "APITimeoutError"},
    )
    test_db_session.add(s_timeout)
    test_db_session.commit()

    metrics = run_token_audit(session=test_db_session)
    assert metrics["runs_count"] == 1
    assert metrics["steps_count"] == 1
    assert metrics["missing_usage_steps"] == 1
    assert metrics["step_total_tokens"] == 0
    assert metrics["step_total_cost_usd"] == 0.0

    captured = capsys.readouterr()
    assert "Missing-Usage Steps     : 1" in captured.out
    assert "failed (UNREC)" in captured.out


def test_audit_supplied_session_does_not_initialize_database(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify that passing an uninitialized session does not mutate schema or create tables."""
    from sqlalchemy import inspect

    # Engine without create_all
    engine = create_engine("sqlite:///:memory:")
    with Session(engine) as uninitialized_session:
        metrics = run_token_audit(session=uninitialized_session)
        assert metrics["runs_count"] == 0
        assert metrics["steps_count"] == 0
        assert metrics["replays_count"] == 0
        # Verify schema inspection: no tables were created
        inspector = inspect(engine)
        assert inspector.get_table_names() == []

    captured = capsys.readouterr()
    assert "Database is currently empty" in captured.out


def test_audit_standalone_fails_cleanly_if_database_not_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that standalone audit cleanly raises FileNotFoundError without creating missing files."""
    from server.app.config import settings

    nonexistent_path = tmp_path / "nonexistent_downshift.db"
    assert not nonexistent_path.exists()

    monkeypatch.setattr(settings, "database_url", f"sqlite:///{nonexistent_path.as_posix()}")

    with pytest.raises(FileNotFoundError, match="does not exist"):
        run_token_audit(session=None)

    # Ensure the file was not inadvertently touched or created
    assert not nonexistent_path.exists()


def test_audit_mixed_successful_and_timed_out_steps_in_run(
    test_db_session: Session, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify mixed successful and timed-out steps clearly separate reconciled cost from unresolved unknown cost."""
    run_mixed = Run(
        id="run-mixed-01",
        task_name="mixed_task",
        agent_name="test_agent",
        baseline_model="nvidia/nemotron-3-ultra-550b-a55b",
        status="failed",
        tokens_in=250,
        tokens_out=100,
        total_cost_usd=0.00035,
        meta={"has_unreconciled_cost": True, "unreconciled_reason": "Timeout on step 2"},
    )
    test_db_session.add(run_mixed)

    # Step 0: Successful Ultra call ($0.00025)
    s0 = Step(
        id="step-m-0",
        run_id="run-mixed-01",
        idx=0,
        tag="plan",
        kind="text",
        model="nvidia/nemotron-3-ultra-550b-a55b",
        tokens_in=100,
        tokens_out=50,
        cost_usd=0.00025,
        response={"usage": {"prompt_tokens": 100, "completion_tokens": 50}},
    )
    # Step 1: Successful Nano call ($0.00010)
    s1 = Step(
        id="step-m-1",
        run_id="run-mixed-01",
        idx=1,
        tag="tool",
        kind="tool_call",
        model="nvidia/nemotron-3-nano-30b-a3b",
        tokens_in=150,
        tokens_out=50,
        cost_usd=0.00010,
        response={"usage": {"prompt_tokens": 150, "completion_tokens": 50}},
    )
    # Step 2: Timed out step (recorded as 0 tokens, $0.00, but unresolved)
    s2 = Step(
        id="step-m-2",
        run_id="run-mixed-01",
        idx=2,
        tag="tool",
        kind="error",
        model="nvidia/nemotron-3-super-120b-a12b",
        tokens_in=0,
        tokens_out=0,
        cost_usd=0.0,
        response={"error": "Request timed out", "error_type": "APITimeoutError"},
    )
    test_db_session.add(s0)
    test_db_session.add(s1)
    test_db_session.add(s2)

    # Replay 0: Successful replay
    r0 = Replay(
        id="replay-m-0",
        step_id="step-m-0",
        tier="super",
        model="nvidia/nemotron-3-super-120b-a12b",
        tokens_in=100,
        tokens_out=40,
        cost_usd=0.00006,
        response={"usage": {"prompt_tokens": 100, "completion_tokens": 40}},
    )
    # Replay 1: Timed out replay
    r1 = Replay(
        id="replay-m-1",
        step_id="step-m-1",
        tier="nano",
        model="nvidia/nemotron-3-nano-30b-a3b",
        tokens_in=0,
        tokens_out=0,
        cost_usd=0.0,
        response={"error": "Replay timed out", "error_type": "APITimeoutError"},
    )
    test_db_session.add(r0)
    test_db_session.add(r1)
    test_db_session.commit()

    metrics = run_token_audit(session=test_db_session)

    # Metrics verification
    assert metrics["runs_count"] == 1
    assert metrics["steps_count"] == 3
    assert metrics["missing_usage_steps"] == 1
    assert metrics["has_unresolved_step_costs"] is True
    assert metrics["step_reconciled_cost_usd"] == pytest.approx(0.00035)

    assert metrics["replays_count"] == 2
    assert metrics["missing_usage_replays"] == 1
    assert metrics["has_unresolved_replay_costs"] is True
    assert metrics["replay_reconciled_cost_usd"] == pytest.approx(0.00006)

    captured = capsys.readouterr()
    assert "[INCOMPLETE - 1 step(s) unresolved]" in captured.out
    assert "Unresolved Steps Cost" in captured.out
    assert "1 call(s) timed out or missing usage; NOT confirmed $0.00" in captured.out
    assert "PARTIAL ONLY - provider may have billed unresolved calls" in captured.out
    assert "[INCOMPLETE: 1 unrec replay(s)]" in captured.out
    assert "Unresolved Replays Cost" in captured.out

"""Tests for Phase 2 step 2a replay engine, mock client, CLI, and cost accounting."""

from pathlib import Path
from typing import Any

import pytest
from sqlmodel import Session, col, select

from server.app import cli
from server.app import db as db_mod
from server.app.config import settings
from server.app.db import create_db_and_tables
from server.app.models import Evaluation, ModelPrice, Replay, Run, Step
from server.app.replay import (
    MockReplayClient,
    _is_retryable_exception,
    _MockChoice,
    _MockMessage,
    _MockResponse,
    _MockUsage,
    replay_run,
    resolve_tier_model,
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


def _create_sample_run(num_steps: int = 3) -> str:
    """Helper to create a sample Run with multiple Steps in database."""
    with Session(db_mod.engine) as session:
        run_record = Run(
            task_name="research_test",
            agent_name="research_agent_test",
            baseline_model="nvidia/Nemotron-3-Ultra-550b-a55b",
            status="completed",
        )
        session.add(run_record)
        session.commit()
        session.refresh(run_record)

        for i in range(num_steps):
            request_data: dict[str, Any] = {
                "messages": [{"role": "user", "content": f"Step question {i}"}],
                "temperature": 0.2,
                "max_tokens": 1500,
                "model": "nvidia/Nemotron-3-Ultra-550b-a55b",
            }
            if i == 1:
                request_data["tools"] = [
                    {
                        "type": "function",
                        "function": {
                            "name": "tavily_search",
                            "parameters": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                            },
                        },
                    }
                ]
                request_data["tool_choice"] = "auto"

            step = Step(
                run_id=run_record.id,
                idx=i,
                tag=f"tag_{i}",
                kind="tool_call" if "tools" in request_data else "text",
                model="nvidia/Nemotron-3-Ultra-550b-a55b",
                request=request_data,
                tokens_in=150,
                tokens_out=60,
                cost_usd=0.000330,
            )
            session.add(step)
        session.commit()
        return run_record.id


def test_replay_rows_stored_and_attributes() -> None:
    """Test that replay rows are created and stored with mode='step_level' and full attributes."""
    run_id = _create_sample_run(num_steps=2)
    client = MockReplayClient()

    replays = replay_run(run_id=run_id, tiers=["super", "nano"], client=client)
    assert len(replays) == 4

    with Session(db_mod.engine) as session:
        db_replays = session.exec(select(Replay)).all()
        matching = [r for r in db_replays if r.step and r.step.run_id == run_id]
        assert len(matching) == 4

        for r in matching:
            assert r.mode == "step_level"
            assert r.tier in ("super", "nano")
            assert "nemotron" in r.model.lower()
            assert r.tokens_in == 150
            assert r.tokens_out == 60
            assert r.latency_ms > 0
            assert r.cost_usd > 0
            assert r.error is None
            assert r.response is not None
            assert "choices" in r.response
            assert len(r.response["choices"]) > 0
            # Reasoning content is preserved in raw response
            msg = r.response["choices"][0]["message"]
            assert msg.get("reasoning_content") is not None


def test_replay_stored_kwargs_reused_exactly_and_only_model_changes() -> None:
    """Test that all request kwargs (temperature, max_tokens, tools, tool_choice) are preserved exactly and only model changes."""
    run_id = _create_sample_run(num_steps=1)

    # Set custom unique kwargs on step
    with Session(db_mod.engine) as session:
        step = session.exec(select(Step).where(Step.run_id == run_id)).first()
        assert step is not None
        step.request = {
            "messages": [{"role": "user", "content": "Special prompt"}],
            "temperature": 0.85,
            "max_tokens": 2468,
            "tools": [{"type": "function", "function": {"name": "custom_tool"}}],
            "tool_choice": "required",
            "model": "nvidia/Nemotron-3-Ultra-550b-a55b",
            "custom_metadata_flag": True,
        }
        session.add(step)
        session.commit()

    captured_kwargs: list[dict[str, Any]] = []

    class TrackingMockClient(MockReplayClient):
        def create(self, *args: Any, **kwargs: Any) -> _MockResponse:
            captured_kwargs.append(kwargs)
            return super().create(*args, **kwargs)

    client = TrackingMockClient()
    replay_run(run_id=run_id, tiers=["super"], client=client)

    assert len(captured_kwargs) == 1
    call_kw = captured_kwargs[0]

    # Stored kwargs reused exactly
    assert call_kw["temperature"] == 0.85
    assert call_kw["max_tokens"] == 2468
    assert call_kw["tools"] == [{"type": "function", "function": {"name": "custom_tool"}}]
    assert call_kw["tool_choice"] == "required"
    assert call_kw["messages"] == [{"role": "user", "content": "Special prompt"}]
    assert call_kw["custom_metadata_flag"] is True

    # ONLY model changes
    with Session(db_mod.engine) as session:
        _, expected_super_id = resolve_tier_model("super", session)
    assert call_kw["model"] == expected_super_id
    assert call_kw["model"] != "nvidia/Nemotron-3-Ultra-550b-a55b"


def test_replay_errors_stored_on_failure() -> None:
    """Test that a failed replay is stored with error and never counts as a pass."""
    run_id = _create_sample_run(num_steps=1)
    client = MockReplayClient(
        tier_behaviors={"nano": RuntimeError("Simulated API failure on Nano")}
    )

    replays = replay_run(run_id=run_id, tiers=["nano"], client=client)
    assert len(replays) == 1
    rep = replays[0]

    assert rep.error is not None
    assert "RuntimeError" in rep.error
    assert "Simulated API failure on Nano" in rep.error
    assert rep.response is None
    assert rep.tool_calls is None
    assert rep.cost_usd == 0.0

    # Persisted in DB
    with Session(db_mod.engine) as session:
        db_rep = session.get(Replay, rep.id)
        assert db_rep is not None
        assert db_rep.error == rep.error


def test_replay_cost_ceiling_abort() -> None:
    """Test that replay aborts before executing calls if cost ceiling is reached."""
    run_id = _create_sample_run(num_steps=3)
    client = MockReplayClient()

    with pytest.raises(RuntimeError, match=r"Cost ceiling of \$0\.000050 reached"):
        replay_run(
            run_id=run_id,
            tiers=["super", "nano"],
            client=client,
            cost_ceiling_usd=0.00005,
            max_workers=1,
        )


def test_replay_idempotent_skip_and_force() -> None:
    """Test that existing replays are skipped unless force=True."""
    run_id = _create_sample_run(num_steps=1)
    client = MockReplayClient()

    # First run creates 1 replay
    replays1 = replay_run(run_id=run_id, tiers=["super"], client=client)
    assert len(replays1) == 1
    first_id = replays1[0].id

    # Second run with force=False skips existing
    replays2 = replay_run(run_id=run_id, tiers=["super"], client=client, force=False)
    assert len(replays2) == 0

    # Third run with force=True re-executes
    replays3 = replay_run(run_id=run_id, tiers=["super"], client=client, force=True)
    assert len(replays3) == 1
    assert replays3[0].id != first_id


def test_replay_idempotent_skip_retries_failed_replays() -> None:
    """Test that existing replays with errors are NOT skipped and are retried on the next run."""
    run_id = _create_sample_run(num_steps=1)

    # 1. First run: client raises error on super tier
    failing_client = MockReplayClient(
        tier_behaviors={"super": RuntimeError("Temporary connection failure")}
    )
    replays1 = replay_run(run_id=run_id, tiers=["super"], client=failing_client)
    assert len(replays1) == 1
    assert replays1[0].error is not None
    failed_id = replays1[0].id

    # 2. Second run with force=False and a working client:
    # Because error IS NOT NULL, it must NOT be skipped, but retried!
    working_client = MockReplayClient()
    replays2 = replay_run(run_id=run_id, tiers=["super"], client=working_client, force=False)
    assert len(replays2) == 1
    assert replays2[0].error is None

    # Check DB: old failed replay replaced by successful replay
    with Session(db_mod.engine) as session:
        step = session.exec(select(Step).where(Step.run_id == run_id)).first()
        assert step is not None
        replays_in_db = session.exec(
            select(Replay).where(Replay.step_id == step.id, Replay.tier == "super")
        ).all()
        assert len(replays_in_db) == 1
        assert replays_in_db[0].error is None
        assert replays_in_db[0].id != failed_id

    # 3. Third run with force=False:
    # Now that error IS NULL, it must be skipped!
    replays3 = replay_run(run_id=run_id, tiers=["super"], client=working_client, force=False)
    assert len(replays3) == 0


def test_replay_force_deletes_evaluation_first() -> None:
    """Test that force=True deletes old replay's Evaluation first, then the replay."""
    run_id = _create_sample_run(num_steps=1)
    client = MockReplayClient()

    # Initial replay
    replays = replay_run(run_id=run_id, tiers=["super"], client=client)
    assert len(replays) == 1
    old_replay_id = replays[0].id

    # Add an Evaluation row attached to this replay
    with Session(db_mod.engine) as session:
        evaluation = Evaluation(
            replay_id=old_replay_id,
            objective_pass=True,
            judge_score=0.95,
            judge_reason="Accurate response",
            judge_model="nemotron_ultra",
        )
        session.add(evaluation)
        session.commit()
        session.refresh(evaluation)
        old_eval_id = evaluation.id

    # Now force=True replay: must delete evaluation first, then replay, and succeed
    new_replays = replay_run(run_id=run_id, tiers=["super"], client=client, force=True)
    assert len(new_replays) == 1
    new_replay_id = new_replays[0].id
    assert new_replay_id != old_replay_id

    with Session(db_mod.engine) as session:
        assert session.get(Replay, old_replay_id) is None
        assert session.get(Evaluation, old_eval_id) is None
        assert session.get(Replay, new_replay_id) is not None


def test_is_retryable_exception_max_tokens_not_retried() -> None:
    """Test that errors with 'max_tokens 1500' in message are NOT retried, while status codes and specific class names are."""
    # 1. Error containing 'max_tokens 1500' must NOT be retryable
    non_retryable_err = ValueError("Invalid parameter: max_tokens 1500 exceeds model context")
    assert not _is_retryable_exception(non_retryable_err)

    # 2. Status codes in (429, 500, 502, 503, 504) ARE retryable
    for code in (429, 500, 502, 503, 504):

        class HttpError(Exception):
            status_code = code

        assert _is_retryable_exception(HttpError(f"HTTP error {code}"))

    # Status codes outside target list are NOT retryable
    class HttpError400(Exception):
        status_code = 400

    assert not _is_retryable_exception(HttpError400("Bad Request"))

    # 3. Class names containing RateLimit, InternalServer, APIConnection or Timeout ARE retryable
    class CustomRateLimitError(Exception):
        pass

    class InternalServerError(Exception):
        pass

    class APIConnectionError(Exception):
        pass

    class APITimeoutError(Exception):
        pass

    for cls in (CustomRateLimitError, InternalServerError, APIConnectionError, APITimeoutError):
        assert _is_retryable_exception(cls("An error occurred"))

    # 4. Verify in replay execution: non-retryable error is called only once
    run_id = _create_sample_run(num_steps=1)
    call_count = 0

    def non_retryable_create(*args: Any, **kwargs: Any) -> _MockResponse:
        nonlocal call_count
        call_count += 1
        raise ValueError("Invalid parameter: max_tokens 1500 exceeds context")

    client = MockReplayClient(tier_behaviors={"super": non_retryable_create})
    replays = replay_run(run_id=run_id, tiers=["super"], client=client)
    assert len(replays) == 1
    assert replays[0].error is not None
    assert "max_tokens 1500" in replays[0].error
    assert call_count == 1  # Called only once, NOT retried!


def test_retry_waits_configurable() -> None:
    """Test that retry waits are configurable and tests can pass tiny waits."""
    assert settings.replay_retry_min_wait == 1.0
    assert settings.replay_retry_max_wait == 30.0
    assert settings.replay_retry_attempts == 4

    run_id = _create_sample_run(num_steps=1)
    call_count = 0

    class RateLimitErr(Exception):
        pass

    def flaky_create(*args: Any, **kwargs: Any) -> _MockResponse:
        nonlocal call_count
        call_count += 1
        if call_count < 4:
            raise RateLimitErr("Rate limit exceeded")
        return _MockResponse(
            model="nvidia/nemotron-3-super-120b-a12b",
            usage=_MockUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            choices=[],
        )

    client = MockReplayClient(tier_behaviors={"super": flaky_create})
    # Pass tiny wait settings directly to replay_run
    replays = replay_run(
        run_id=run_id,
        tiers=["super"],
        client=client,
        retry_min_wait=0.001,
        retry_max_wait=0.005,
        retry_multiplier=0.001,
        retry_attempts=4,
    )
    assert len(replays) == 1
    assert replays[0].error is None
    assert call_count == 4


def test_replay_run_fails_if_prices_yaml_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that a missing prices.yaml halts replay_run with an explicit error."""
    run_id = _create_sample_run(num_steps=1)
    monkeypatch.setattr(settings, "prices_path", tmp_path / "nonexistent_prices.yaml")

    with pytest.raises(FileNotFoundError):
        replay_run(run_id=run_id, tiers=["super"])


def test_replay_max_steps_limit() -> None:
    """Test that max_steps limits replay to the first N steps of a run."""
    run_id = _create_sample_run(num_steps=4)
    client = MockReplayClient()

    replays = replay_run(run_id=run_id, tiers=["super"], client=client, max_steps=2)
    assert len(replays) == 2

    with Session(db_mod.engine) as session:
        steps = session.exec(
            select(Step).where(Step.run_id == run_id).order_by(col(Step.idx))
        ).all()
        replayed_step_ids = {r.step_id for r in replays}
        assert steps[0].id in replayed_step_ids
        assert steps[1].id in replayed_step_ids
        assert steps[2].id not in replayed_step_ids
        assert steps[3].id not in replayed_step_ids


def test_replay_cost_uses_tier_own_price() -> None:
    """Test that replay cost is calculated using the candidate tier's own pricing."""
    run_id = _create_sample_run(num_steps=1)
    client = MockReplayClient(prompt_tokens=150, completion_tokens=60)

    replays = replay_run(run_id=run_id, tiers=["super", "nano"], client=client)
    assert len(replays) == 2

    super_rep = next(r for r in replays if r.tier == "super")
    nano_rep = next(r for r in replays if r.tier == "nano")

    # Super rates: $0.30 in / $0.90 out -> (150*0.30 + 60*0.90)/1M = $0.000099
    assert super_rep.cost_usd == pytest.approx(0.000099)

    # Nano rates: $0.06 in / $0.24 out -> (150*0.06 + 60*0.24)/1M = $0.0000234
    assert nano_rep.cost_usd == pytest.approx(0.0000234)

    # Baseline Ultra cost for same tokens is $0.000330 -> tiers are strictly cheaper
    assert super_rep.cost_usd < 0.000330
    assert nano_rep.cost_usd < super_rep.cost_usd


def test_mock_replay_client_deterministic_per_tier() -> None:
    """Test MockReplayClient per-tier deterministic behavior (Nano returns wrong tool arg, Super is correct)."""
    client = MockReplayClient()

    tools_kw = {
        "tools": [{"type": "function", "function": {"name": "search"}}],
        "tool_choice": "auto",
    }

    # Nano tool call
    nano_resp = client.create(model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B", **tools_kw)
    nano_tc = nano_resp.choices[0].message.tool_calls
    assert nano_tc is not None
    assert "wrong_nano_argument" in nano_tc[0].function.arguments

    # Super tool call
    super_resp = client.create(model="nvidia/nemotron-3-super-120b-a12b", **tools_kw)
    super_tc = super_resp.choices[0].message.tool_calls
    assert super_tc is not None
    assert "NVIDIA Nemotron architecture specifications" in super_tc[0].function.arguments


def test_replay_tenacity_retry_on_429_5xx() -> None:
    """Test that tenacity retries on simulated 429/5xx and succeeds after recovery."""
    run_id = _create_sample_run(num_steps=1)
    call_count = 0

    class MockRateLimitError(Exception):
        status_code = 429

    def flaky_create(*args: Any, **kwargs: Any) -> _MockResponse:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise MockRateLimitError("429 Too Many Requests (Rate limit exceeded)")
        return _MockResponse(
            model="nvidia/nemotron-3-super-120b-a12b",
            usage=_MockUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            choices=[],
        )

    client = MockReplayClient(tier_behaviors={"super": flaky_create})
    replays = replay_run(
        run_id=run_id,
        tiers=["super"],
        client=client,
        retry_min_wait=0.001,
        retry_max_wait=0.005,
    )

    assert len(replays) == 1
    assert replays[0].error is None
    assert call_count == 3


def test_cli_replay_mock_and_real_refusal(capsys: pytest.CaptureFixture[str]) -> None:
    """Test CLI replay in default mock mode and refusal when --real is passed."""
    run_id = _create_sample_run(num_steps=1)

    # 1. Test refusal with --real
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["replay", run_id, "--tiers", "super,nano", "--real"])
    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert (
        "REFUSAL: Real API calls are disabled in Phase 2 Step 2a. Replays only run in mock mode."
        in err
    )

    # 2. Test successful mock replay via CLI
    cli.main(["replay", run_id, "--tiers", "super,nano"])
    out = capsys.readouterr().out
    assert f"Downshift Replay Summary: Run {run_id}" in out
    assert "Total Replays Generated : 2" in out


def test_retrying_failed_step_tier_deletes_old_replay_and_evaluation() -> None:
    """Retrying a failed (step, tier): delete old errored Replay and Evaluation before inserting new row, ensuring exactly one row."""
    run_id = _create_sample_run(num_steps=1)

    # 1. First run: client raises error, producing a failed replay
    failing_client = MockReplayClient(tier_behaviors={"super": RuntimeError("Simulated failure")})
    failed_replays = replay_run(run_id=run_id, tiers=["super"], client=failing_client)
    assert len(failed_replays) == 1
    assert failed_replays[0].error is not None
    old_replay_id = failed_replays[0].id

    # 2. Attach an Evaluation row to the failed replay
    with Session(db_mod.engine) as session:
        old_eval = Evaluation(
            replay_id=old_replay_id,
            objective_pass=False,
            judge_score=0.0,
            judge_reason="Prior run failed with runtime error",
            judge_model="nemotron_ultra",
        )
        session.add(old_eval)
        session.commit()
        session.refresh(old_eval)
        old_eval_id = old_eval.id

        # Verify DB initially has 1 replay and 1 evaluation
        step = session.exec(select(Step).where(Step.run_id == run_id)).first()
        assert step is not None
        replays_before = session.exec(
            select(Replay).where(Replay.step_id == step.id, Replay.tier == "super")
        ).all()
        assert len(replays_before) == 1

    # 3. Second run retrying the failed step (force=False) with a working client
    working_client = MockReplayClient()
    new_replays = replay_run(run_id=run_id, tiers=["super"], client=working_client, force=False)
    assert len(new_replays) == 1
    assert new_replays[0].error is None
    new_replay_id = new_replays[0].id
    assert new_replay_id != old_replay_id

    # 4. Confirm in DB: old errored replay and its evaluation are deleted, exactly one replay row exists
    with Session(db_mod.engine) as session:
        all_replays = session.exec(
            select(Replay).where(Replay.step_id == step.id, Replay.tier == "super")
        ).all()
        assert len(all_replays) == 1
        assert all_replays[0].id == new_replay_id
        assert all_replays[0].error is None

        # Old replay and old evaluation are gone
        assert session.get(Replay, old_replay_id) is None
        assert session.get(Evaluation, old_eval_id) is None


def test_replay_missing_usage_or_token_counts_stored_as_error_and_zero_cost() -> None:
    """If a response has no usage, or no prompt_tokens or completion_tokens, store with error and cost 0, never success."""
    run_id = _create_sample_run(num_steps=1)

    # Case A: Response has no usage (usage=None)
    def no_usage_create(*args: Any, **kwargs: Any) -> _MockResponse:
        return _MockResponse(
            model="nvidia/nemotron-3-super-120b-a12b",
            usage=None,
            choices=[
                _MockChoice(message=_MockMessage(role="assistant", content="Answer without usage"))
            ],
        )

    client_a = MockReplayClient(tier_behaviors={"super": no_usage_create})
    replays_a = replay_run(run_id=run_id, tiers=["super"], client=client_a)
    assert len(replays_a) == 1
    assert replays_a[0].error is not None
    assert "no usage" in replays_a[0].error.lower()
    assert replays_a[0].cost_usd == 0.0
    assert replays_a[0].tokens_in == 0
    assert replays_a[0].tokens_out == 0

    # Case B: Response has usage but no prompt_tokens (prompt_tokens=None)
    def no_prompt_tokens_create(*args: Any, **kwargs: Any) -> _MockResponse:
        return _MockResponse(
            model="nvidia/nemotron-3-super-120b-a12b",
            usage=_MockUsage(prompt_tokens=None, completion_tokens=50),
            choices=[
                _MockChoice(
                    message=_MockMessage(role="assistant", content="Answer without prompt_tokens")
                )
            ],
        )

    client_b = MockReplayClient(tier_behaviors={"super": no_prompt_tokens_create})
    replays_b = replay_run(run_id=run_id, tiers=["super"], client=client_b, force=True)
    assert len(replays_b) == 1
    assert replays_b[0].error is not None
    assert "prompt_tokens" in replays_b[0].error.lower()
    assert replays_b[0].cost_usd == 0.0

    # Case C: Response has usage but no completion_tokens (completion_tokens=None)
    def no_completion_tokens_create(*args: Any, **kwargs: Any) -> _MockResponse:
        return _MockResponse(
            model="nvidia/nemotron-3-super-120b-a12b",
            usage=_MockUsage(prompt_tokens=100, completion_tokens=None),
            choices=[
                _MockChoice(
                    message=_MockMessage(
                        role="assistant", content="Answer without completion_tokens"
                    )
                )
            ],
        )

    client_c = MockReplayClient(tier_behaviors={"super": no_completion_tokens_create})
    replays_c = replay_run(run_id=run_id, tiers=["super"], client=client_c, force=True)
    assert len(replays_c) == 1
    assert replays_c[0].error is not None
    assert "completion_tokens" in replays_c[0].error.lower()
    assert replays_c[0].cost_usd == 0.0


def test_sqlite_wal_and_busy_timeout_concurrent_replays() -> None:
    """SQLite WAL mode and busy timeout on engine allows max_workers=4 and 20 mock replays without 'database is locked'."""
    # Verify connect event pragmas on engine
    with db_mod.engine.connect() as conn:
        journal_mode = conn.exec_driver_sql("PRAGMA journal_mode").scalar()
        busy_timeout = conn.exec_driver_sql("PRAGMA busy_timeout").scalar()

    if "memory" not in str(db_mod.engine.url):
        assert str(journal_mode).lower() == "wal"
    assert busy_timeout == 5000

    # 20 steps replayed with max_workers=4
    run_id = _create_sample_run(num_steps=20)
    client = MockReplayClient()

    replays = replay_run(
        run_id=run_id,
        tiers=["super"],
        client=client,
        max_workers=4,
    )
    assert len(replays) == 20
    assert all(r.error is None for r in replays)

    with Session(db_mod.engine) as session:
        stored_replays = session.exec(select(Replay).join(Step).where(Step.run_id == run_id)).all()
        assert len(stored_replays) == 20


def test_replay_run_refuses_live_client_when_pricing_unverified() -> None:
    """Replay run refuses to execute when passed a live OpenAI client and pricing is unverified."""
    from openai import OpenAI

    live_client = OpenAI(api_key="mock-test-key", base_url="https://api.mock.nebius/v1")
    run_id = _create_sample_run(num_steps=1)

    with pytest.raises(
        RuntimeError, match="REFUSAL: Cannot execute replay against live provider client"
    ):
        replay_run(run_id=run_id, tiers=["super"], client=live_client)


def test_replay_run_refuses_adversarial_client_types_when_pricing_unverified() -> None:
    """Replay run refuses execution for subclassed, proxied, or mock-wrapped live clients."""
    from openai import OpenAI

    run_id = _create_sample_run(num_steps=1)

    # 1. Subclass of OpenAI
    class SubclassOpenAI(OpenAI):
        pass

    sub_client = SubclassOpenAI(api_key="mock", base_url="https://api.mock.nebius/v1")
    with pytest.raises(
        RuntimeError, match="REFUSAL: Cannot execute replay against live provider client"
    ):
        replay_run(run_id=run_id, tiers=["super"], client=sub_client)

    # 2. Proxy client that wraps OpenAI
    class ProxyClient:
        def __init__(self) -> None:
            self.client = OpenAI(api_key="mock", base_url="https://api.mock.nebius/v1")
            self.chat = self.client.chat

    proxy_client = ProxyClient()
    with pytest.raises(
        RuntimeError, match="REFUSAL: Cannot execute replay against live provider client"
    ):
        replay_run(run_id=run_id, tiers=["super"], client=proxy_client)

    # 3. Mock class name that maliciously wraps an actual live client
    class MockTrojanWrapper:
        def __init__(self) -> None:
            self.client = OpenAI(api_key="mock", base_url="https://api.mock.nebius/v1")
            self.chat = self.client.chat

    trojan_client = MockTrojanWrapper()
    with pytest.raises(
        RuntimeError, match="REFUSAL: Cannot execute replay against live provider client"
    ):
        replay_run(run_id=run_id, tiers=["super"], client=trojan_client)

    # 4. Third-party duck-typed client
    class ThirdPartyOpenAICompatible:
        def __init__(self) -> None:
            self.chat = self
            self.completions = self

        def create(self, **_kwargs: Any) -> Any:
            raise NotImplementedError("Live call")

    third_party_client = ThirdPartyOpenAICompatible()
    with pytest.raises(
        RuntimeError, match="REFUSAL: Cannot execute replay against live provider client"
    ):
        replay_run(run_id=run_id, tiers=["super"], client=third_party_client)

    # 5. Setting is_mock=True on a live OpenAI client cannot bypass the gate
    tricky_client = OpenAI(api_key="mock", base_url="https://api.mock.nebius/v1")
    tricky_client.is_mock = True  # type: ignore[attr-defined]
    with pytest.raises(
        RuntimeError, match="REFUSAL: Cannot execute replay against live provider client"
    ):
        replay_run(run_id=run_id, tiers=["super"], client=tricky_client)

    # 6. Subclassing MockReplayClient while wrapping an inner live client cannot bypass the gate
    class SneakySubclassMock(MockReplayClient):
        def __init__(self) -> None:
            super().__init__()
            self.hidden_real_handle = OpenAI(api_key="mock", base_url="https://api.mock.nebius/v1")

    sneaky_client = SneakySubclassMock()
    with pytest.raises(
        RuntimeError, match="REFUSAL: Cannot execute replay against live provider client"
    ):
        replay_run(run_id=run_id, tiers=["super"], client=sneaky_client)

    # 7. Slotted proxy hiding a live OpenAI client cannot bypass the gate
    class SlottedReplayProxy:
        __slots__ = ("_hidden_client",)

        def __init__(self) -> None:
            self._hidden_client = OpenAI(api_key="mock", base_url="https://api.mock.nebius/v1")

    slotted_client = SlottedReplayProxy()
    with pytest.raises(
        RuntimeError, match="REFUSAL: Cannot execute replay against live provider client"
    ):
        replay_run(run_id=run_id, tiers=["super"], client=slotted_client)

    # 8. Spoofing _mock_self or _mock_return_value attributes cannot bypass the gate
    class SpoofedMockAttributeReplayClient:
        def __init__(self) -> None:
            self._mock_self = True

    spoofed_attr_client = SpoofedMockAttributeReplayClient()
    with pytest.raises(
        RuntimeError, match="REFUSAL: Cannot execute replay against live provider client"
    ):
        replay_run(run_id=run_id, tiers=["super"], client=spoofed_attr_client)

    # 9. Spoofing __name__ and __module__ as MockReplayClient cannot bypass the gate
    class SpoofedReplayIdentityClient:
        pass

    SpoofedReplayIdentityClient.__name__ = "MockReplayClient"
    SpoofedReplayIdentityClient.__module__ = "server.app.replay"
    spoofed_replay_client = SpoofedReplayIdentityClient()
    with pytest.raises(
        RuntimeError, match="REFUSAL: Cannot execute replay against live provider client"
    ):
        replay_run(run_id=run_id, tiers=["super"], client=spoofed_replay_client)

    # 10. Verify execution_mode gate refusal when execution_mode is offline (default)
    with pytest.raises(RuntimeError, match="execution_mode is 'offline'"):
        replay_run(run_id=run_id, tiers=["super"], client=sub_client, execution_mode="offline")

    # 11. Verify execution_mode="live" still refuses when live pricing is unverified
    with pytest.raises(
        RuntimeError,
        match="REFUSAL: Cannot execute replay against live provider client: prices.yaml has verified_for_live=false",
    ):
        replay_run(run_id=run_id, tiers=["super"], client=sub_client, execution_mode="live")


def test_replay_run_clamps_max_tokens() -> None:
    """Replay run ensures max_tokens is bounded: defaults to 600 if missing, clamps if oversized."""
    captured_kwargs: list[dict[str, Any]] = []

    def mock_create(kwargs: dict[str, Any]) -> _MockResponse:
        captured_kwargs.append(copy.deepcopy(kwargs))
        return _MockResponse(
            model=kwargs["model"],
            usage=_MockUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            choices=[
                _MockChoice(
                    message=_MockMessage(role="assistant", content="Replayed text"),
                    finish_reason="stop",
                )
            ],
        )

    import copy

    # Step 1 has no max_tokens; Step 2 has max_tokens=10000 (oversized)
    with Session(db_mod.engine) as session:
        run_record = Run(
            task_name="clamp_test",
            agent_name="agent_clamp",
            baseline_model="nvidia/Nemotron-3-Ultra-550b-a55b",
            status="completed",
        )
        session.add(run_record)
        session.commit()
        session.refresh(run_record)

        s1 = Step(
            run_id=run_record.id,
            idx=0,
            tag="plan",
            kind="text",
            model="nvidia/Nemotron-3-Ultra-550b-a55b",
            request={"model": "nvidia/Nemotron-3-Ultra-550b-a55b", "messages": []},
            tokens_in=10,
            tokens_out=10,
            cost_usd=0.0001,
        )
        s2 = Step(
            run_id=run_record.id,
            idx=1,
            tag="synthesis",
            kind="text",
            model="nvidia/Nemotron-3-Ultra-550b-a55b",
            request={
                "model": "nvidia/Nemotron-3-Ultra-550b-a55b",
                "messages": [],
                "max_tokens": 10000,
            },
            tokens_in=10,
            tokens_out=10,
            cost_usd=0.0001,
        )
        session.add_all([s1, s2])
        session.commit()
        run_id = run_record.id

    client = MockReplayClient(tier_behaviors={"super": mock_create})
    replays = replay_run(run_id=run_id, tiers=["super"], client=client, force=True)
    assert len(replays) == 2
    assert all(r.error is None for r in replays)

    assert len(captured_kwargs) == 2
    # First step had no max_tokens -> defaulted to 600
    assert captured_kwargs[0]["max_tokens"] == 600
    # Second step had 10000 -> clamped to 4096
    assert captured_kwargs[1]["max_tokens"] == 4096

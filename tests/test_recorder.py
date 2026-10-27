"""Tests for Downshift recorder SDK."""

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel
from sqlmodel import Session, select

import server.app.db as db_mod
from sdk import downshift
from server.app.models import ModelPrice, Run, Step


class MockUsage(BaseModel):
    """Mock usage object."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class MockMessage(BaseModel):
    """Mock message object."""

    role: str = "assistant"
    content: str = "Test response"
    tool_calls: list[dict[str, str]] | None = None


class MockChoice(BaseModel):
    """Mock choice object."""

    message: MockMessage
    index: int = 0


class MockResponse(BaseModel):
    """Mock completion response."""

    id: str = "chatcmpl-test"
    model: str = "nemotron_ultra"
    usage: MockUsage
    choices: list[MockChoice]


def create_mock_client(
    model: str = "nemotron_ultra", prompt_tokens: int = 100, completion_tokens: int = 50
) -> MagicMock:
    """Create a mock OpenAI-compatible client."""
    client = MagicMock()
    mock_resp = MockResponse(
        model=model,
        usage=MockUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        choices=[MockChoice(message=MockMessage())],
    )
    client.chat.completions.create.return_value = mock_resp
    return client


def test_step_stored_and_tag_stripped() -> None:
    """Test that steps are recorded, tags stripped, and response returned untouched."""
    with Session(db_mod.engine) as session:
        session.add(
            ModelPrice(
                model="nemotron_ultra",
                input_per_mtok=1.0,
                output_per_mtok=2.0,
            )
        )
        session.commit()

    raw_client = create_mock_client()
    wrapped_client = downshift.wrap_client(raw_client)

    with downshift.run(task_name="test_task", agent_name="test_agent") as active_run:
        res = wrapped_client.chat.completions.create(
            model="nemotron_ultra",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=256,
            temperature=0.7,
            downshift_tag="search_query",
        )

    # Tag was stripped before forwarding to target client
    raw_client.chat.completions.create.assert_called_once()
    _, kwargs = raw_client.chat.completions.create.call_args
    assert "downshift_tag" not in kwargs
    assert res.id == "chatcmpl-test"

    # Step record in database
    with Session(db_mod.engine) as session:
        run_record = session.get(Run, active_run.id)
        assert run_record is not None
        assert run_record.status == "completed"

        steps = session.exec(select(Step).where(Step.run_id == active_run.id)).all()
        assert len(steps) == 1
        assert steps[0].tag == "search_query"
        assert steps[0].idx == 0
        assert steps[0].tokens_in == 100
        assert steps[0].tokens_out == 50
        assert steps[0].request["max_tokens"] == 256
        assert steps[0].request["temperature"] == 0.7
        assert "downshift_tag" not in steps[0].request
        # Cost: (100 * 1.0 + 50 * 2.0) / 1,000,000 = 0.0002
        assert pytest.approx(steps[0].cost_usd, 1e-9) == 0.0002


def test_exact_denylist_strips_secrets_and_keeps_allowed_params() -> None:
    """Test exact denylist (api_key, authorization, headers, extra_headers) and retention of allowed params."""
    with Session(db_mod.engine) as session:
        session.add(
            ModelPrice(
                model="nemotron_ultra",
                input_per_mtok=1.0,
                output_per_mtok=2.0,
            )
        )
        session.commit()

    raw_client = create_mock_client()
    wrapped_client = downshift.wrap_client(raw_client)

    with downshift.run(task_name="test_task", agent_name="test_agent") as active_run:
        wrapped_client.chat.completions.create(
            model="nemotron_ultra",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=100,
            max_completion_tokens=150,
            tool_choice="auto",
            api_key="secret-api-key",
            authorization="Bearer secret-token",
            headers={"X-Custom": "secret"},
            extra_headers={"X-Extra": "secret"},
        )

    with Session(db_mod.engine) as session:
        steps = session.exec(select(Step).where(Step.run_id == active_run.id)).all()
        req = steps[0].request
        # Exact denylist must be stripped
        assert "api_key" not in req
        assert "authorization" not in req
        assert "headers" not in req
        assert "extra_headers" not in req
        # Allowed parameters must be kept
        assert req["max_tokens"] == 100
        assert req["max_completion_tokens"] == 150
        assert req["tool_choice"] == "auto"


def test_totals_equal_sum_of_step_costs() -> None:
    """Test that run totals equal the sum of step tokens and costs."""
    with Session(db_mod.engine) as session:
        session.add(
            ModelPrice(
                model="nemotron_ultra",
                input_per_mtok=2.5,
                output_per_mtok=5.0,
            )
        )
        session.commit()

    raw_client = create_mock_client()
    wrapped_client = downshift.wrap_client(raw_client)

    with downshift.run(task_name="multi_step_task", agent_name="test_agent") as active_run:
        wrapped_client.chat.completions.create(
            model="nemotron_ultra",
            messages=[{"role": "user", "content": "step 1"}],
            max_tokens=128,
            downshift_tag="step_one",
        )
        wrapped_client.chat.completions.create(
            model="nemotron_ultra",
            messages=[{"role": "user", "content": "step 2"}],
            max_tokens=128,
            downshift_tag="step_two",
        )

    with Session(db_mod.engine) as session:
        run_record = session.get(Run, active_run.id)
        assert run_record is not None
        steps = session.exec(select(Step).where(Step.run_id == active_run.id)).all()
        assert len(steps) == 2

        sum_tokens_in = sum(s.tokens_in for s in steps)
        sum_tokens_out = sum(s.tokens_out for s in steps)
        sum_cost = sum(s.cost_usd for s in steps)

        assert run_record.tokens_in == sum_tokens_in
        assert run_record.tokens_out == sum_tokens_out
        assert pytest.approx(run_record.total_cost_usd, 1e-9) == sum_cost


def test_preflight_null_price_fails_before_any_api_call() -> None:
    """Test that preflight check raises an error and makes 0 API calls if baseline price is null."""
    with Session(db_mod.engine) as session:
        session.add(
            ModelPrice(
                model="nemotron_ultra",
                input_per_mtok=None,
                output_per_mtok=None,
            )
        )
        session.commit()

    raw_client = create_mock_client()
    wrapped_client = downshift.wrap_client(raw_client)

    with (
        pytest.raises(ValueError, match="Preflight check failed"),
        downshift.run(task_name="null_price_task", agent_name="test_agent"),
    ):
        wrapped_client.chat.completions.create(
            model="nemotron_ultra",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=50,
        )

    raw_client.chat.completions.create.assert_not_called()


def test_keyboard_interrupt_leaves_status_failed() -> None:
    """Test that KeyboardInterrupt leaves run status as failed."""
    with Session(db_mod.engine) as session:
        session.add(
            ModelPrice(
                model="nemotron_ultra",
                input_per_mtok=1.0,
                output_per_mtok=2.0,
            )
        )
        session.commit()

    run_id = None
    with (
        pytest.raises(KeyboardInterrupt),
        downshift.run(task_name="interrupt_task", agent_name="test_agent") as active_run,
    ):
        run_id = active_run.id
        raise KeyboardInterrupt("Simulated user interrupt")

    with Session(db_mod.engine) as session:
        run_record = session.get(Run, run_id)
        assert run_record is not None
        assert run_record.status == "failed"


def test_missing_max_tokens_rejected() -> None:
    """Test that omitting both max_tokens and max_completion_tokens raises ValueError."""
    with Session(db_mod.engine) as session:
        session.add(
            ModelPrice(
                model="nemotron_ultra",
                input_per_mtok=1.0,
                output_per_mtok=2.0,
            )
        )
        session.commit()

    raw_client = create_mock_client()
    wrapped_client = downshift.wrap_client(raw_client)

    with (
        pytest.raises(ValueError, match="max_tokens must be explicitly set"),
        downshift.run(task_name="no_max_tokens", agent_name="test_agent"),
    ):
        wrapped_client.chat.completions.create(
            model="nemotron_ultra",
            messages=[{"role": "user", "content": "hello"}],
        )


def test_cost_ceiling_aborts_before_call() -> None:
    """Test that exceeding per-run cost ceiling raises RuntimeError before making the call."""
    with Session(db_mod.engine) as session:
        session.add(
            ModelPrice(
                model="nemotron_ultra",
                input_per_mtok=100.0,
                output_per_mtok=200.0,
            )
        )
        session.commit()

    raw_client = create_mock_client(prompt_tokens=5000, completion_tokens=5000)
    wrapped_client = downshift.wrap_client(raw_client)

    # Cost for 5k/5k at $100/$200 is $1.50
    # Set ceiling of $1.00: call 1 makes total $1.50 (which brings total >= ceiling).
    # Call 2 must be aborted before calling the target client.
    with downshift.run(
        task_name="ceiling_task",
        agent_name="test_agent",
        cost_ceiling_usd=1.00,
    ):
        wrapped_client.chat.completions.create(
            model="nemotron_ultra",
            messages=[{"role": "user", "content": "call 1"}],
            max_tokens=100,
        )
        assert raw_client.chat.completions.create.call_count == 1

        with pytest.raises(RuntimeError, match="Cost ceiling .* reached"):
            wrapped_client.chat.completions.create(
                model="nemotron_ultra",
                messages=[{"role": "user", "content": "call 2"}],
                max_tokens=100,
            )

        # Ensure call 2 was never forwarded to the client
        assert raw_client.chat.completions.create.call_count == 1


# --- Tests for the three new record_call fixes ---


def test_price_lookup_requested_first_and_fallback() -> None:
    """Test price lookup tries requested model ID first, then falls back to response.model."""
    with Session(db_mod.engine) as session:
        session.add(
            ModelPrice(
                model="requested_model_id",
                input_per_mtok=1.0,
                output_per_mtok=2.0,
            )
        )
        session.add(
            ModelPrice(
                model="fallback_response_model_id",
                input_per_mtok=3.0,
                output_per_mtok=4.0,
            )
        )
        session.commit()

    # Case 1: Requested model is known -> matched to requested model
    client_1 = MagicMock()
    client_1.chat.completions.create.return_value = MockResponse(
        model="unconfigured_server_model",
        usage=MockUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        choices=[MockChoice(message=MockMessage())],
    )
    wrapped_1 = downshift.wrap_client(client_1)
    with downshift.run(
        task_name="lookup_task_1", agent_name="test_agent", baseline_model="requested_model_id"
    ) as r1:
        wrapped_1.chat.completions.create(
            model="requested_model_id",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=50,
        )

    with Session(db_mod.engine) as session:
        steps_1 = session.exec(select(Step).where(Step.run_id == r1.id)).all()
        assert len(steps_1) == 1
        assert steps_1[0].model == "requested_model_id"

    # Case 2: Requested model is unknown/alias that doesn't resolve -> falls back to response.model
    client_2 = MagicMock()
    client_2.chat.completions.create.return_value = MockResponse(
        model="fallback_response_model_id",
        usage=MockUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        choices=[MockChoice(message=MockMessage())],
    )
    wrapped_2 = downshift.wrap_client(client_2)
    with downshift.run(
        task_name="lookup_task_2",
        agent_name="test_agent",
        baseline_model="fallback_response_model_id",
    ) as r2:
        wrapped_2.chat.completions.create(
            model="unconfigured_requested_model",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=50,
        )

    with Session(db_mod.engine) as session:
        steps_2 = session.exec(select(Step).where(Step.run_id == r2.id)).all()
        assert len(steps_2) == 1
        assert steps_2[0].model == "fallback_response_model_id"


def test_forty_step_limit_enforced() -> None:
    """Test that attempting to execute more than 40 steps raises RuntimeError."""
    with Session(db_mod.engine) as session:
        session.add(
            ModelPrice(
                model="nemotron_ultra",
                input_per_mtok=0.1,
                output_per_mtok=0.2,
            )
        )
        session.commit()

    raw_client = create_mock_client()
    wrapped_client = downshift.wrap_client(raw_client)

    with downshift.run(task_name="limit_test", agent_name="test_agent"):
        for i in range(40):
            wrapped_client.chat.completions.create(
                model="nemotron_ultra",
                messages=[{"role": "user", "content": f"msg {i}"}],
                max_tokens=10,
            )

        # 41st call must be blocked
        with pytest.raises(RuntimeError, match="Maximum step limit \\(40\\) reached"):
            wrapped_client.chat.completions.create(
                model="nemotron_ultra",
                messages=[{"role": "user", "content": "step 41"}],
                max_tokens=10,
            )


def test_failed_call_counts_toward_step_limit() -> None:
    """Test that failed call attempts count toward the 40-step limit before the call."""
    with Session(db_mod.engine) as session:
        session.add(
            ModelPrice(
                model="nemotron_ultra",
                input_per_mtok=0.1,
                output_per_mtok=0.2,
            )
        )
        session.commit()

    failing_client = MagicMock()
    failing_client.chat.completions.create.side_effect = RuntimeError("Simulated API failure")
    wrapped_client = downshift.wrap_client(failing_client)

    with downshift.run(task_name="fail_limit_test", agent_name="test_agent"):
        # Make 40 calls that fail
        for i in range(40):
            with pytest.raises(RuntimeError, match="Simulated API failure"):
                wrapped_client.chat.completions.create(
                    model="nemotron_ultra",
                    messages=[{"role": "user", "content": f"msg {i}"}],
                    max_tokens=10,
                )

        assert failing_client.chat.completions.create.call_count == 40

        # Attempt 41 must be aborted by the step limit BEFORE calling the client
        with pytest.raises(RuntimeError, match="Maximum step limit \\(40\\) reached"):
            wrapped_client.chat.completions.create(
                model="nemotron_ultra",
                messages=[{"role": "user", "content": "step 41"}],
                max_tokens=10,
            )

        # Call count remains 40 (call 41 was never forwarded)
        assert failing_client.chat.completions.create.call_count == 40


def test_prospective_cost_check_blocks_call_before_dispatch() -> None:
    """Test that prospective cost check prevents call dispatch when projected max cost exceeds budget."""
    with Session(db_mod.engine) as session:
        session.add(
            ModelPrice(
                model="nemotron_ultra",
                input_per_mtok=1.0,
                output_per_mtok=3.0,
            )
        )
        session.commit()

    raw_client = create_mock_client()
    wrapped_client = downshift.wrap_client(raw_client)

    # Budget is $0.0010; a call with max_tokens=1000 projected output cost is $0.0030
    with (
        pytest.raises(RuntimeError, match="prospective call cost"),
        downshift.run(
            task_name="budget_task",
            agent_name="test_agent",
            baseline_model="nemotron_ultra",
            cost_ceiling_usd=0.0010,
        ),
    ):
        wrapped_client.chat.completions.create(
            model="nemotron_ultra",
            messages=[{"role": "user", "content": "Request that would exceed budget"}],
            max_tokens=1000,
        )

    # Zero network/mock calls were dispatched
    assert raw_client.chat.completions.create.call_count == 0


def test_missing_usage_metadata_marks_run_unreconciled_and_fails() -> None:
    """Test that a response with missing usage metadata marks run as failed with unreconciled cost."""
    with Session(db_mod.engine) as session:
        session.add(
            ModelPrice(
                model="nemotron_ultra",
                input_per_mtok=1.0,
                output_per_mtok=3.0,
            )
        )
        session.commit()

    # Mock response without usage attribute
    response_no_usage = MagicMock()
    response_no_usage.usage = None
    response_no_usage.model = "nemotron_ultra"
    response_no_usage.choices = [MagicMock(message=MagicMock(content="answer", tool_calls=None))]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = response_no_usage
    wrapped_client = downshift.wrap_client(mock_client)

    run_id: str | None = None
    with (
        pytest.raises(ValueError, match="Response usage metadata is absent or malformed"),
        downshift.run(task_name="missing_usage_task", agent_name="test_agent") as active_run,
    ):
        run_id = active_run.id
        wrapped_client.chat.completions.create(
            model="nemotron_ultra",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=50,
        )

    assert run_id is not None
    with Session(db_mod.engine) as session:
        run_record = session.get(Run, run_id)
        assert run_record is not None
        assert run_record.status == "failed"
        assert run_record.meta is not None
        assert run_record.meta.get("has_unreconciled_cost") is True
        assert "unreconciled_reason" in run_record.meta


def test_unreconciled_cost_blocks_subsequent_calls_in_run() -> None:
    """Test that once a run encounters unreconciled cost, subsequent calls in that run are blocked."""
    with Session(db_mod.engine) as session:
        session.add(
            ModelPrice(
                model="nemotron_ultra",
                input_per_mtok=1.0,
                output_per_mtok=3.0,
            )
        )
        session.commit()

    response_no_usage = MagicMock()
    response_no_usage.usage = None
    response_no_usage.model = "nemotron_ultra"

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = response_no_usage
    wrapped_client = downshift.wrap_client(mock_client)

    with downshift.run(task_name="blocked_subsequent_task", agent_name="test_agent"):
        # First call fails on usage metadata
        with pytest.raises(ValueError, match="Response usage metadata is absent or malformed"):
            wrapped_client.chat.completions.create(
                model="nemotron_ultra",
                messages=[{"role": "user", "content": "call 1"}],
                max_tokens=50,
            )

        # Second call must be blocked before calling client
        with pytest.raises(RuntimeError, match="has unreconciled costs from a previous call"):
            wrapped_client.chat.completions.create(
                model="nemotron_ultra",
                messages=[{"role": "user", "content": "call 2"}],
                max_tokens=50,
            )

    # Client was only called once (for call 1); call 2 was blocked before dispatch
    assert mock_client.chat.completions.create.call_count == 1


def test_no_active_run_context_raises_clear_error() -> None:
    """Test that calling wrapped client outside downshift.run context raises RuntimeError."""
    raw_client = create_mock_client()
    wrapped_client = downshift.wrap_client(raw_client)

    with pytest.raises(
        RuntimeError,
        match="No active Downshift run context. Calls to wrapped client must be made inside",
    ):
        wrapped_client.chat.completions.create(
            model="nemotron_ultra",
            messages=[{"role": "user", "content": "outside context"}],
            max_tokens=20,
        )


def test_tool_definitions_increase_prospective_cost_and_block_dispatch() -> None:
    """Test that tool definitions increase prospective input cost and block call before dispatch when budget is tight."""
    with Session(db_mod.engine) as session:
        session.add(
            ModelPrice(
                model="nemotron_ultra",
                input_per_mtok=100.0,
                output_per_mtok=10.0,
            )
        )
        session.commit()

    raw_client = create_mock_client()
    wrapped_client = downshift.wrap_client(raw_client)

    # Budget is $0.0100.
    # Without tools: input is minimal (hi + 50 framing tokens = 50 tokens = $0.0050 input + $0.0001 output = $0.0051), fits budget.
    with downshift.run(
        task_name="budget_fit_test", agent_name="test_agent", cost_ceiling_usd=0.0100
    ):
        wrapped_client.chat.completions.create(
            model="nemotron_ultra",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=10,
        )
    assert raw_client.chat.completions.create.call_count == 1

    # Reset client mock
    raw_client.chat.completions.create.reset_mock()

    # With a large tool definition (~2000 chars):
    # Tool adds ~500+ tokens (~$0.0500+ input cost), pushing prospective cost to > $0.0550, exceeding $0.0100 budget.
    large_tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup_database_records",
                "description": "A" * 2000,
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    with (
        pytest.raises(RuntimeError, match="prospective call cost"),
        downshift.run(
            task_name="budget_blocked_test",
            agent_name="test_agent",
            cost_ceiling_usd=0.0100,
        ),
    ):
        wrapped_client.chat.completions.create(
            model="nemotron_ultra",
            messages=[{"role": "user", "content": "hi"}],
            tools=large_tools,
            max_tokens=10,
        )

    # Verification: client was never called because call was blocked pre-dispatch
    assert raw_client.chat.completions.create.call_count == 0


def test_malformed_tool_definition_handled_safely() -> None:
    """Test that malformed or non-dict tool structures do not crash prospective estimation."""
    with Session(db_mod.engine) as session:
        session.add(
            ModelPrice(
                model="nemotron_ultra",
                input_per_mtok=1.0,
                output_per_mtok=1.0,
            )
        )
        session.commit()

    raw_client = create_mock_client()
    wrapped_client = downshift.wrap_client(raw_client)

    class CustomNonSerializableTool:
        def __repr__(self) -> str:
            return "CustomNonSerializableToolObject"

    with downshift.run(task_name="malformed_tool_task", agent_name="test_agent"):
        res = wrapped_client.chat.completions.create(
            model="nemotron_ultra",
            messages=[{"role": "user", "content": "hi"}],
            tools=[CustomNonSerializableTool()],
            max_tokens=10,
        )
    assert res.id == "chatcmpl-test"
    assert raw_client.chat.completions.create.call_count == 1


def test_concurrent_calls_respect_budget_ceiling() -> None:
    """Test that concurrent threads sharing a run context cannot race past the cost ceiling."""
    import threading

    from sdk.downshift.recorder import RunContext

    with Session(db_mod.engine) as session:
        session.add(
            ModelPrice(
                model="nemotron_ultra",
                input_per_mtok=100.0,
                output_per_mtok=100.0,
            )
        )
        session.commit()

    with Session(db_mod.engine) as session:
        run = Run(
            task_name="concurrent_test",
            agent_name="test_agent",
            baseline_model="nemotron_ultra",
        )
        session.add(run)
        session.commit()
        session.refresh(run)

    # Ceiling is $0.0500. Each call prospective cost is $0.0200.
    # At most 2 calls can be reserved simultaneously; 3 calls would be $0.0600 > $0.0500.
    run_ctx = RunContext(run, cost_ceiling_usd=0.0500)
    prospective_cost = 0.0200

    results: list[tuple[int | None, Exception | None]] = []
    lock = threading.Lock()

    def try_reserve() -> None:
        try:
            idx = run_ctx.check_and_reserve(prospective_cost)
            with lock:
                results.append((idx, None))
        except RuntimeError as e:
            with lock:
                results.append((None, e))

    threads = [threading.Thread(target=try_reserve) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    successes = [r for r in results if r[0] is not None]
    failures = [r for r in results if r[1] is not None]

    assert len(successes) == 2
    assert len(failures) == 3
    for f in failures:
        assert "exceeds budget" in str(f[1])


def test_timeout_outcome_records_error_and_blocks_subsequent_calls() -> None:
    """A timeout outcome must be recorded as an error step, mark run unreconciled, and block further calls."""
    from openai import APITimeoutError
    from sqlmodel import select

    with Session(db_mod.engine) as session:
        session.add(
            ModelPrice(
                model="nemotron_ultra",
                input_per_mtok=1.0,
                output_per_mtok=3.0,
            )
        )
        session.commit()

    failing_client = MagicMock()
    failing_client.chat.completions.create.side_effect = APITimeoutError(request=MagicMock())
    wrapped_client = downshift.wrap_client(failing_client)

    with downshift.run(
        task_name="timeout_unrec_test", agent_name="test_agent", baseline_model="nemotron_ultra"
    ) as run_record:
        # 1. First call times out
        with pytest.raises(APITimeoutError):
            wrapped_client.chat.completions.create(
                model="nemotron_ultra",
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=50,
            )

        # 2. Verify Step in database is kind='error' with 0 cost
        with Session(db_mod.engine) as session:
            steps = session.exec(select(Step).where(Step.run_id == run_record.id)).all()
            assert len(steps) == 1
            assert steps[0].kind == "error"
            assert steps[0].tokens_in == 0
            assert steps[0].tokens_out == 0
            assert steps[0].cost_usd == 0.0

            db_run = session.get(Run, run_record.id)
            assert db_run is not None
            assert db_run.status == "failed"
            assert db_run.meta is not None
            assert db_run.meta.get("has_unreconciled_cost") is True
            assert "Call timed out" in db_run.meta.get("unreconciled_reason", "")

        # 3. Second call in same run must be immediately blocked before dispatch
        with pytest.raises(RuntimeError, match="has unreconciled costs from a previous call"):
            wrapped_client.chat.completions.create(
                model="nemotron_ultra",
                messages=[{"role": "user", "content": "second try"}],
                max_tokens=50,
            )

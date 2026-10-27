"""Tests for Phase 2 step 2b objective checks, evaluation saving, and CLI."""

from typing import Any

import pytest
from sqlmodel import Session, select

from server.app import cli
from server.app import db as db_mod
from server.app.checks import check_replay, evaluate_replay, evaluate_run
from server.app.db import create_db_and_tables
from server.app.models import Evaluation, ModelPrice, Replay, Run, Step
from server.app.replay import MockReplayClient, replay_run


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


def _create_test_step(
    kind: str = "tool_call",
    tools: list[dict[str, Any]] | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
) -> Step:
    """Helper to build a transient Step instance with configured request and tools."""
    if tools is None:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "tavily_search",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "max_results": {"type": "integer"},
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            }
        ]

    if tool_calls is None and kind == "tool_call":
        tool_calls = [
            {
                "id": "call_orig_1",
                "type": "function",
                "function": {
                    "name": "tavily_search",
                    "arguments": '{"query": "nemotron specifications"}',
                },
            }
        ]

    return Step(
        id="step_test_1",
        run_id="run_test_1",
        idx=0,
        tag="research_query",
        kind=kind,
        model="nvidia/Nemotron-3-Ultra-550b-a55b",
        request={"model": "nvidia/Nemotron-3-Ultra-550b-a55b", "tools": tools},
        tool_calls=tool_calls,
        tokens_in=100,
        tokens_out=50,
        cost_usd=0.00025,
    )


def _create_test_replay(
    error: str | None = None,
    finish_reason: str = "tool_calls",
    tool_calls: list[dict[str, Any]] | None = None,
    content: str | None = None,
    reasoning_content: str | None = None,
) -> Replay:
    """Helper to build a transient Replay instance with custom response payload."""
    msg: dict[str, Any] = {"role": "assistant"}
    if content is not None:
        msg["content"] = content
    if reasoning_content is not None:
        msg["reasoning_content"] = reasoning_content
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls

    response_payload = {
        "id": "cmpl_replay_test",
        "model": "nvidia/nemotron-3-super-120b-a12b",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": msg,
            }
        ],
        "usage": {"prompt_tokens": 80, "completion_tokens": 40, "total_tokens": 120},
    }

    return Replay(
        id="replay_test_1",
        step_id="step_test_1",
        tier="super",
        model="nvidia/nemotron-3-super-120b-a12b",
        mode="step_level",
        response=response_payload,
        tool_calls=tool_calls,
        tokens_in=80,
        tokens_out=40,
        latency_ms=120.0,
        cost_usd=0.00006,
        error=error,
    )


# 1. Test: Wrong tool name
def test_check_replay_wrong_tool_name() -> None:
    """If replay calls a different tool name, tool_names_and_order fails and objective_pass is False."""
    step = _create_test_step(kind="tool_call")
    replay = _create_test_replay(
        finish_reason="tool_calls",
        tool_calls=[
            {
                "id": "call_rep_1",
                "type": "function",
                "function": {
                    "name": "wrong_search_tool",
                    "arguments": '{"query": "nemotron specs"}',
                },
            }
        ],
    )

    passed, checks = check_replay(step, replay)
    assert passed is False
    check_map = {c["name"]: c for c in checks}
    assert check_map["tool_names_and_order"]["passed"] is False
    assert "Tool names or order differ" in check_map["tool_names_and_order"]["reason"]


# 2. Test: Invalid JSON
def test_check_replay_invalid_json() -> None:
    """If replay tool arguments do not parse as valid JSON, arguments_valid_json fails."""
    step = _create_test_step(kind="tool_call")
    replay = _create_test_replay(
        finish_reason="tool_calls",
        tool_calls=[
            {
                "id": "call_rep_1",
                "type": "function",
                "function": {
                    "name": "tavily_search",
                    "arguments": "NOT_VALID_JSON{query: broken",
                },
            }
        ],
    )

    passed, checks = check_replay(step, replay)
    assert passed is False
    check_map = {c["name"]: c for c in checks}
    assert check_map["arguments_valid_json"]["passed"] is False
    assert "failed JSON parse" in check_map["arguments_valid_json"]["reason"]


# 3. Test: Missing required key
def test_check_replay_missing_required_key() -> None:
    """If replay tool arguments are missing a required schema key, arguments_schema_validation fails."""
    step = _create_test_step(kind="tool_call")
    replay = _create_test_replay(
        finish_reason="tool_calls",
        tool_calls=[
            {
                "id": "call_rep_1",
                "type": "function",
                "function": {
                    "name": "tavily_search",
                    "arguments": '{"max_results": 5}',  # Missing required 'query'
                },
            }
        ],
    )

    passed, checks = check_replay(step, replay)
    assert passed is False
    check_map = {c["name"]: c for c in checks}
    assert check_map["arguments_schema_validation"]["passed"] is False
    assert (
        "missing required parameter 'query'" in check_map["arguments_schema_validation"]["reason"]
    )


# 4. Test: Unknown key with additionalProperties false
def test_check_replay_unknown_key_additional_properties_false() -> None:
    """Unknown keys fail only when additionalProperties is false."""
    # Tool has additionalProperties: False
    step = _create_test_step(kind="tool_call")
    replay = _create_test_replay(
        finish_reason="tool_calls",
        tool_calls=[
            {
                "id": "call_rep_1",
                "type": "function",
                "function": {
                    "name": "tavily_search",
                    "arguments": '{"query": "valid query", "unexpected_extra": 123}',
                },
            }
        ],
    )

    passed, checks = check_replay(step, replay)
    assert passed is False
    check_map = {c["name"]: c for c in checks}
    assert check_map["arguments_schema_validation"]["passed"] is False
    assert "additionalProperties is false" in check_map["arguments_schema_validation"]["reason"]

    # When additionalProperties is omitted (or True), unexpected keys are permitted
    loose_tools = [
        {
            "type": "function",
            "function": {
                "name": "tavily_search",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    ]
    loose_step = _create_test_step(kind="tool_call", tools=loose_tools)
    passed_loose, _ = check_replay(loose_step, replay)
    assert passed_loose is True


# 5. Test: Replay error
def test_check_replay_replay_error() -> None:
    """Any replay with a recorded error fails objective check immediately."""
    step = _create_test_step(kind="tool_call")
    replay = _create_test_replay(error="RuntimeError: 500 Internal Server Error")

    passed, checks = check_replay(step, replay)
    assert passed is False
    check_map = {c["name"]: c for c in checks}
    assert check_map["no_replay_error"]["passed"] is False
    assert "500 Internal Server Error" in check_map["no_replay_error"]["reason"]


# 6. Test: Length ending
def test_check_replay_length_ending() -> None:
    """For a text step, finish_reason 'length' fails objective checks."""
    step = _create_test_step(kind="text", tools=[])
    replay = _create_test_replay(
        finish_reason="length",
        content="This answer was cut off prematurely due to max_token limits...",
    )

    passed, checks = check_replay(step, replay)
    assert passed is False
    check_map = {c["name"]: c for c in checks}
    assert check_map["finish_reason_stop"]["passed"] is False
    assert "length ending fails" in check_map["finish_reason_stop"]["reason"]


# 7. Test: Empty content
def test_check_replay_empty_content() -> None:
    """For a text step, empty content string fails content_non_empty."""
    step = _create_test_step(kind="text", tools=[])
    replay = _create_test_replay(
        finish_reason="stop",
        content="   ",  # Blank whitespace
    )

    passed, checks = check_replay(step, replay)
    assert passed is False
    check_map = {c["name"]: c for c in checks}
    assert check_map["content_non_empty"]["passed"] is False
    assert "Response content is empty" in check_map["content_non_empty"]["reason"]


# 8. Test: Reasoning-only content
def test_check_replay_reasoning_only_content() -> None:
    """An answer that exists only in a reasoning field (reasoning_content or reasoning) fails."""
    step = _create_test_step(kind="text", tools=[])
    replay = _create_test_replay(
        finish_reason="stop",
        content="",
        reasoning_content="Here is the step-by-step reasoning that should have gone to content.",
    )

    passed, checks = check_replay(step, replay)
    assert passed is False
    check_map = {c["name"]: c for c in checks}
    assert check_map["content_non_empty"]["passed"] is False
    assert "Answer exists only in reasoning field" in check_map["content_non_empty"]["reason"]


# 9. Test: All-pass case (tool_call with differing query values, and clean text step)
def test_check_replay_all_pass_case() -> None:
    """Verify clean tool_call and text steps pass completely, and argument differences are info only."""
    # Tool call step: query value is legitimately different ("nebius models" vs "nemotron specifications")
    step_tool = _create_test_step(kind="tool_call")
    replay_tool = _create_test_replay(
        finish_reason="tool_calls",
        tool_calls=[
            {
                "id": "call_rep_1",
                "type": "function",
                "function": {
                    "name": "tavily_search",
                    "arguments": '{"query": "nebius models", "max_results": 3}',
                },
            }
        ],
    )

    passed_tool, checks_tool = check_replay(step_tool, replay_tool)
    assert passed_tool is True
    check_map_tool = {c["name"]: c for c in checks_tool}
    assert check_map_tool["finish_reason_tool_calls"]["passed"] is True
    assert check_map_tool["tool_names_and_order"]["passed"] is True
    assert check_map_tool["arguments_valid_json"]["passed"] is True
    assert check_map_tool["arguments_schema_validation"]["passed"] is True
    assert check_map_tool["arguments_equality_info"]["passed"] is True
    assert check_map_tool["arguments_equality_info"]["arguments_equal"] is False
    assert "differ legitimately" in check_map_tool["arguments_equality_info"]["reason"]

    # Text step: non-empty content with finish_reason 'stop'
    step_text = _create_test_step(kind="text", tools=[])
    replay_text = _create_test_replay(
        finish_reason="stop",
        content="Comprehensive answer covering essential facts.",
    )
    passed_text, checks_text = check_replay(step_text, replay_text)
    assert passed_text is True
    check_map_text = {c["name"]: c for c in checks_text}
    assert check_map_text["content_non_empty"]["passed"] is True
    assert check_map_text["finish_reason_stop"]["passed"] is True


# 10. Test: Idempotency and Evaluation Saving
def test_evaluate_run_idempotency_and_storage() -> None:
    """Save result as one Evaluation row per replay (judge fields stay empty). Re-running must not create duplicates."""
    # 1. Setup a real Run and Step in DB
    with Session(db_mod.engine) as session:
        run_record = Run(
            task_name="eval_test",
            agent_name="agent_test",
            baseline_model="nvidia/Nemotron-3-Ultra-550b-a55b",
            status="completed",
        )
        session.add(run_record)
        session.commit()
        session.refresh(run_record)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "tavily_search",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ]
        step = Step(
            run_id=run_record.id,
            idx=0,
            tag="search_query",
            kind="tool_call",
            model="nvidia/Nemotron-3-Ultra-550b-a55b",
            request={"model": "nvidia/Nemotron-3-Ultra-550b-a55b", "tools": tools},
            tool_calls=[
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "tavily_search", "arguments": '{"query": "orig query"}'},
                }
            ],
            tokens_in=100,
            tokens_out=50,
            cost_usd=0.00025,
        )
        session.add(step)
        session.commit()
        run_id = run_record.id

    # 2. Run replay on candidate tiers (MockReplayClient)
    client = MockReplayClient()
    replays = replay_run(run_id=run_id, tiers=["super", "nano"], client=client)
    assert len(replays) == 2

    # 3. First evaluation run
    evals_first = evaluate_run(run_id)
    assert len(evals_first) == 2

    with Session(db_mod.engine) as session:
        all_evals = session.exec(select(Evaluation)).all()
        assert len(all_evals) == 2
        for ev in all_evals:
            # Judge fields stay empty
            assert ev.judge_score is None
            assert ev.judge_reason is None
            assert ev.judge_model is None
            assert ev.objective_detail is not None
            assert "checks" in ev.objective_detail

    # 4. Re-running evaluation must not create duplicates
    evals_second = evaluate_run(run_id)
    assert len(evals_second) == 2

    with Session(db_mod.engine) as session:
        all_evals_after = session.exec(select(Evaluation)).all()
        assert len(all_evals_after) == 2  # Exactly 2 rows, no duplicates!


# 11. Test: CLI evaluate
def test_cli_evaluate(capsys: pytest.CaptureFixture[str]) -> None:
    """Test python -m server.app.cli evaluate <run_id> prints pass rate per tag and tier."""
    with Session(db_mod.engine) as session:
        run_record = Run(
            task_name="cli_eval_test",
            agent_name="agent_test",
            baseline_model="nvidia/Nemotron-3-Ultra-550b-a55b",
            status="completed",
        )
        session.add(run_record)
        session.commit()
        session.refresh(run_record)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "tavily_search",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ]
        step = Step(
            run_id=run_record.id,
            idx=0,
            tag="search_query",
            kind="tool_call",
            model="nvidia/Nemotron-3-Ultra-550b-a55b",
            request={"model": "nvidia/Nemotron-3-Ultra-550b-a55b", "tools": tools},
            tool_calls=[
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "tavily_search", "arguments": '{"query": "orig query"}'},
                }
            ],
            tokens_in=100,
            tokens_out=50,
            cost_usd=0.00025,
        )
        session.add(step)
        session.commit()
        run_id = run_record.id

    client = MockReplayClient()
    replay_run(run_id=run_id, tiers=["super", "nano"], client=client)

    cli.main(["evaluate", run_id])
    out = capsys.readouterr().out

    assert f"Downshift Objective Evaluation Summary: Run {run_id}" in out
    assert "Tag" in out
    assert "Tier" in out
    assert "Pass Rate" in out
    assert "search_query" in out
    assert "super" in out
    assert "nano" in out


# 12. Test: Replayed tool matches original step but has no schema in request tools
def test_check_replay_matching_tool_name_missing_schema_in_request() -> None:
    """If replayed tool matches original step name but no schema is in step.request tools, schema validation fails."""
    # Original step calls 'custom_tool', but tools in request only defines 'other_tool'
    step = _create_test_step(
        kind="tool_call",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "other_tool",
                    "parameters": {"type": "object", "properties": {"x": {"type": "string"}}},
                },
            }
        ],
        tool_calls=[
            {
                "id": "call_orig_1",
                "type": "function",
                "function": {"name": "custom_tool", "arguments": '{"query": "test"}'},
            }
        ],
    )
    replay = _create_test_replay(
        finish_reason="tool_calls",
        tool_calls=[
            {
                "id": "call_rep_1",
                "type": "function",
                "function": {"name": "custom_tool", "arguments": '{"query": "test"}'},
            }
        ],
    )

    passed, checks = check_replay(step, replay)
    assert passed is False
    check_map = {c["name"]: c for c in checks}
    assert check_map["tool_names_and_order"]["passed"] is True  # Names matched!
    assert check_map["arguments_schema_validation"]["passed"] is False
    assert (
        "Schema definition for tool 'custom_tool' not found in original request tools"
        in check_map["arguments_schema_validation"]["reason"]
    )


# 13. Test: Re-evaluating existing Evaluation updates objective fields without overwriting judge fields
def test_evaluate_replay_preserves_existing_judge_fields() -> None:
    """Verify that re-evaluating an existing Evaluation updates objective fields without overwriting judge-related fields."""
    step = _create_test_step(kind="text", tools=[])
    replay = _create_test_replay(
        finish_reason="stop",
        content="Valid synthesized final answer.",
    )

    with Session(db_mod.engine) as session:
        # Pre-seed an existing Evaluation row with existing judge data and false objective_pass
        existing_eval = Evaluation(
            replay_id=replay.id,
            objective_pass=False,
            objective_detail={"checks": []},
            judge_score=0.92,
            judge_reason="High quality response verified by LLM judge",
            judge_model="nemotron_ultra",
        )
        session.add(existing_eval)
        session.commit()
        session.refresh(existing_eval)
        eval_id = existing_eval.id

        # Re-evaluate
        updated_eval = evaluate_replay(step, replay, session)
        assert updated_eval.id == eval_id
        # Objective fields are updated to reflect the new evaluation
        assert updated_eval.objective_pass is True
        assert updated_eval.objective_detail is not None
        assert len(updated_eval.objective_detail["checks"]) > 0

        # Judge-related fields remain completely preserved
        assert updated_eval.judge_score == 0.92
        assert updated_eval.judge_reason == "High quality response verified by LLM judge"
        assert updated_eval.judge_model == "nemotron_ultra"

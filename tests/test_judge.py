"""Tests for Phase 2c LLM-based replay quality judging, bias controls, and CLI."""

from typing import Any

import pytest
from sqlmodel import Session, select

from server.app import cli
from server.app import db as db_mod
from server.app.checks import evaluate_replay
from server.app.db import create_db_and_tables
from server.app.judge import (
    MockJudgeClient,
    _build_judge_prompt,
    _objective_checks_have_run,
    judge_replay,
    judge_run,
)
from server.app.models import Evaluation, ModelPrice, Replay, Run, Step


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


def _create_sample_run_with_step(
    kind: str = "text",
    tag: str = "final_synthesis",
    content: str = "Reference comprehensive synthesis of Nemotron token factory capabilities.",
    tool_calls: list[dict[str, Any]] | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> tuple[str, str]:
    """Helper to create a Run and Step in DB, returning (run_id, step_id)."""
    with Session(db_mod.engine) as session:
        run_record = Run(
            task_name="judge_test",
            agent_name="agent_judge_test",
            baseline_model="nvidia/Nemotron-3-Ultra-550b-a55b",
            status="completed",
        )
        session.add(run_record)
        session.commit()
        session.refresh(run_record)

        if tools is None and kind == "tool_call":
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
            tag=tag,
            kind=kind,
            model="nvidia/Nemotron-3-Ultra-550b-a55b",
            request={
                "model": "nvidia/Nemotron-3-Ultra-550b-a55b",
                "tools": tools or [],
                "messages": [
                    {"role": "user", "content": "Research NVIDIA Nemotron models on Nebius."}
                ],
            },
            response={
                "choices": [
                    {
                        "finish_reason": "stop" if kind == "text" else "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": content,
                            "tool_calls": tool_calls,
                        },
                    }
                ]
            },
            tool_calls=tool_calls,
            tokens_in=120,
            tokens_out=60,
            cost_usd=0.0003,
        )
        session.add(step)
        session.commit()
        session.refresh(step)
        return run_record.id, step.id


def _create_sample_replay(
    step_id: str,
    tier: str = "super",
    finish_reason: str = "stop",
    content: str = "Candidate synthesized answer covering essential facts accurately.",
    tool_calls: list[dict[str, Any]] | None = None,
    error: str | None = None,
) -> Replay:
    """Helper to persist a Replay row in DB."""
    with Session(db_mod.engine) as session:
        replay = Replay(
            step_id=step_id,
            tier=tier,
            model=f"nvidia/nemotron-3-{tier}-test",
            mode="step_level",
            response={
                "choices": [
                    {
                        "finish_reason": finish_reason,
                        "message": {
                            "role": "assistant",
                            "content": content,
                            "tool_calls": tool_calls,
                        },
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            },
            tool_calls=tool_calls,
            tokens_in=100,
            tokens_out=50,
            cost_usd=0.00005,
            error=error,
        )
        session.add(replay)
        session.commit()
        session.refresh(replay)
        return replay


# 1. Test: High-quality text replay
def test_judge_high_quality_text() -> None:
    """Judge gives high score and [PASS] prefix to semantically accurate text replay."""
    _run_id, step_id = _create_sample_run_with_step(kind="text", content="Full accurate summary.")
    replay = _create_sample_replay(step_id, content="Accurate candidate summary with key points.")

    with Session(db_mod.engine) as session:
        step = session.get(Step, step_id)
        assert step is not None

        client = MockJudgeClient(
            default_score=0.92,
            default_verdict="pass",
            default_reason="Semantic accuracy and completeness preserved cleanly.",
        )
        ev = judge_replay(step, replay, client=client, session=session)

        assert ev.judge_score == 0.92
        assert ev.judge_reason is not None
        assert ev.judge_reason.startswith("[PASS]")
        assert "Semantic accuracy" in ev.judge_reason
        assert ev.judge_model is not None


# 2. Test: Low-quality / hallucinated text replay
def test_judge_low_quality_text() -> None:
    """Judge gives low score and [FAIL] prefix to degraded or hallucinated text replay."""
    _run_id, step_id = _create_sample_run_with_step(
        kind="text", content="Accurate technical facts."
    )
    replay = _create_sample_replay(
        step_id, content="Completely fabricated information contradicting context."
    )

    with Session(db_mod.engine) as session:
        step = session.get(Step, step_id)
        assert step is not None

        client = MockJudgeClient(
            default_score=0.25,
            default_verdict="fail",
            default_reason="Contradicts reference output and hallucinates unsupported claims.",
        )
        ev = judge_replay(step, replay, client=client, session=session)

        assert ev.judge_score == 0.25
        assert ev.judge_reason is not None
        assert ev.judge_reason.startswith("[FAIL]")


# 3. Test: Tool-call intent preservation with differing query wording
def test_judge_tool_call_intent_preservation() -> None:
    """Judge awards [PASS] when candidate search query uses different wording but preserves research intent."""
    orig_tc = [
        {
            "id": "c1",
            "type": "function",
            "function": {
                "name": "tavily_search",
                "arguments": '{"query": "nvidia nemotron 3 super specifications"}',
            },
        }
    ]
    _run_id, step_id = _create_sample_run_with_step(kind="tool_call", tool_calls=orig_tc)

    candidate_tc = [
        {
            "id": "c_rep",
            "type": "function",
            "function": {
                "name": "tavily_search",
                "arguments": '{"query": "nemotron 3 super 120b architecture specs token factory"}',
            },
        }
    ]
    replay = _create_sample_replay(
        step_id, finish_reason="tool_calls", tool_calls=candidate_tc, content=""
    )

    with Session(db_mod.engine) as session:
        step = session.get(Step, step_id)
        assert step is not None

        client = MockJudgeClient(
            default_score=0.88,
            default_verdict="pass",
            default_reason="Search query rephrases terms while preserving core technical research intent.",
        )
        ev = judge_replay(step, replay, client=client, session=session)

        assert ev.objective_pass is True
        assert ev.judge_score == 0.88
        assert ev.judge_reason is not None
        assert ev.judge_reason.startswith("[PASS]")


# 4. Test: Tool-call divergent intent
def test_judge_tool_call_divergent_intent() -> None:
    """Judge gives low score and [FAIL] when candidate tool query drifts completely away from task goal."""
    orig_tc = [
        {
            "id": "c1",
            "type": "function",
            "function": {
                "name": "tavily_search",
                "arguments": '{"query": "nemotron model pricing"}',
            },
        }
    ]
    _run_id, step_id = _create_sample_run_with_step(kind="tool_call", tool_calls=orig_tc)

    divergent_tc = [
        {
            "id": "c_rep",
            "type": "function",
            "function": {
                "name": "tavily_search",
                "arguments": '{"query": "weather forecast tomorrow"}',
            },
        }
    ]
    replay = _create_sample_replay(
        step_id, finish_reason="tool_calls", tool_calls=divergent_tc, content=""
    )

    with Session(db_mod.engine) as session:
        step = session.get(Step, step_id)
        assert step is not None

        client = MockJudgeClient(
            default_score=0.15,
            default_verdict="fail",
            default_reason="Query has completely drifted from the research topic to unrelated weather search.",
        )
        ev = judge_replay(step, replay, client=client, session=session)

        assert ev.judge_score == 0.15
        assert ev.judge_reason is not None
        assert ev.judge_reason.startswith("[FAIL]")


# 5. Test: Uncertainty with score=None
def test_judge_uncertainty_score_none() -> None:
    """For an uncertain verdict, judge_score must be None, not converted to 0.0, and prefixed with [UNCERTAIN]."""
    _run_id, step_id = _create_sample_run_with_step(kind="text")
    replay = _create_sample_replay(step_id, content="Ambiguous response.")

    with Session(db_mod.engine) as session:
        step = session.get(Step, step_id)
        assert step is not None

        client = MockJudgeClient(
            default_score=None,
            default_verdict="uncertain",
            default_reason="Insufficient context to verify domain-specific claim.",
        )
        ev = judge_replay(step, replay, client=client, session=session)

        # Must be None, not 0.0!
        assert ev.judge_score is None
        assert ev.judge_reason is not None
        assert ev.judge_reason.startswith("[UNCERTAIN]")


# 6. Test: Malformed judge output
def test_judge_malformed_output() -> None:
    """Malformed judge response sets judge_score to None and prefixes with [ERROR], never passing."""
    _run_id, step_id = _create_sample_run_with_step(kind="text")
    replay = _create_sample_replay(step_id)

    with Session(db_mod.engine) as session:
        step = session.get(Step, step_id)
        assert step is not None

        # Handler returns plain unparseable text
        client = MockJudgeClient(
            scenario_handlers={
                "Compare the following": "I cannot parse this into json format sorry!"
            }
        )
        ev = judge_replay(step, replay, client=client, session=session)

        assert ev.judge_score is None
        assert ev.judge_reason is not None
        assert ev.judge_reason.startswith("[ERROR]")
        assert "Malformed judge output" in ev.judge_reason


# 7. Test: Judge exception handling
def test_judge_exception_handling() -> None:
    """Judge client exceptions are recorded as [ERROR] with score None."""
    _run_id, step_id = _create_sample_run_with_step(kind="text")
    replay = _create_sample_replay(step_id)

    with Session(db_mod.engine) as session:
        step = session.get(Step, step_id)
        assert step is not None

        client = MockJudgeClient(
            scenario_handlers={"Compare the following": RuntimeError("Simulated judge API timeout")}
        )
        ev = judge_replay(step, replay, client=client, session=session)

        assert ev.judge_score is None
        assert ev.judge_reason is not None
        assert ev.judge_reason.startswith("[ERROR]")
        assert "Simulated judge API timeout" in ev.judge_reason


# 8. Test: Objective-failure short-circuiting
def test_judge_objective_failure_short_circuiting() -> None:
    """Replays that fail objective checks are short-circuited with score 0.0 without calling judge client."""
    _run_id, step_id = _create_sample_run_with_step(kind="text")
    # Replay with error fails objective checks
    replay = _create_sample_replay(step_id, error="RuntimeError: Model generation failed")

    with Session(db_mod.engine) as session:
        step = session.get(Step, step_id)
        assert step is not None

        # Pre-seed evaluation with objective_pass = False
        ev_obj = evaluate_replay(step, replay, session)
        assert ev_obj.objective_pass is False

        judge_called = False

        def tracking_create(*args: Any, **kwargs: Any) -> Any:
            nonlocal judge_called
            judge_called = True
            raise AssertionError(
                "Judge should NOT have been called for an objectively failed replay!"
            )

        client = MockJudgeClient(scenario_handlers={"Compare": tracking_create})
        ev = judge_replay(step, replay, client=client, session=session)

        assert not judge_called
        assert ev.judge_score == 0.0
        assert ev.judge_reason is not None
        assert ev.judge_reason.startswith("[FAIL]")
        assert "Short-circuited" in ev.judge_reason


# 9. Test: Unevaluated replay evaluates objective checks first
def test_judge_unevaluated_replay_evaluates_objective_first() -> None:
    """An unevaluated replay is not treated as failed; objective checks run first before judging."""
    _run_id, step_id = _create_sample_run_with_step(kind="text", content="Ref answer")
    replay = _create_sample_replay(step_id, content="Candidate answer")

    with Session(db_mod.engine) as session:
        step = session.get(Step, step_id)
        assert step is not None

        # Confirm no Evaluation exists yet in DB
        assert (
            session.exec(select(Evaluation).where(Evaluation.replay_id == replay.id)).first()
            is None
        )

        client = MockJudgeClient(default_score=0.85, default_verdict="pass")
        ev = judge_replay(step, replay, client=client, session=session)

        # Objective check ran and passed
        assert ev.objective_pass is True
        assert ev.objective_detail is not None
        # Judge then ran and scored
        assert ev.judge_score == 0.85
        assert ev.judge_reason is not None
        assert ev.judge_reason.startswith("[PASS]")


# 9b. Regression Test: Existing Evaluation row with objective_pass=False unverified
def test_existing_evaluation_row_with_objective_pass_false_unverified() -> None:
    """An existing Evaluation row with objective_pass=False (e.g. default schema value)
    must not be treated as a genuine objective failure if objective checks have not actually run.
    Instead, objective checks must run first, and if passed, the replay proceeds to the judge.
    """
    _run_id, step_id = _create_sample_run_with_step(kind="text", content="Baseline reference text.")
    # This replay is objectively valid (finish_reason='stop', non-empty content)
    replay = _create_sample_replay(step_id, finish_reason="stop", content="Valid candidate text.")

    with Session(db_mod.engine) as session:
        step = session.get(Step, step_id)
        assert step is not None

        # Pre-seed an Evaluation row with objective_pass=False and objective_detail=None
        # simulating a row created without objective checks having been run
        pre_existing_ev = Evaluation(
            replay_id=replay.id,
            objective_pass=False,
            objective_detail=None,
        )
        session.add(pre_existing_ev)
        session.commit()
        session.refresh(pre_existing_ev)
        assert pre_existing_ev.objective_pass is False
        assert pre_existing_ev.objective_detail is None

        # Verify our helper distinguishes this as unverified
        assert not _objective_checks_have_run(pre_existing_ev)

        client = MockJudgeClient(default_score=0.92, default_verdict="pass")
        ev = judge_replay(step, replay, client=client, session=session)

        # Objective checks must have run now and passed!
        assert ev.id == pre_existing_ev.id
        assert ev.objective_pass is True
        assert _objective_checks_have_run(ev)
        assert ev.objective_detail is not None

        # Judge should have run and NOT been short-circuited with score 0.0
        assert ev.judge_score == 0.92
        assert ev.judge_reason is not None
        assert ev.judge_reason.startswith("[PASS]")
        assert "Short-circuited" not in ev.judge_reason


def test_existing_evaluation_row_with_empty_objective_detail_unverified() -> None:
    """An existing Evaluation row with objective_pass=False and objective_detail={}
    must also be recognized as unverified and evaluated objectively before judging.
    """
    _run_id, step_id = _create_sample_run_with_step(kind="text", content="Baseline reference text.")
    replay = _create_sample_replay(step_id, finish_reason="stop", content="Valid candidate text.")

    with Session(db_mod.engine) as session:
        step = session.get(Step, step_id)
        assert step is not None

        pre_existing_ev = Evaluation(
            replay_id=replay.id,
            objective_pass=False,
            objective_detail={},
        )
        session.add(pre_existing_ev)
        session.commit()
        session.refresh(pre_existing_ev)

        assert not _objective_checks_have_run(pre_existing_ev)

        client = MockJudgeClient(default_score=0.88, default_verdict="pass")
        ev = judge_replay(step, replay, client=client, session=session)

        assert ev.objective_pass is True
        assert ev.judge_score == 0.88
        assert ev.judge_reason is not None
        assert ev.judge_reason.startswith("[PASS]")
        assert "Short-circuited" not in ev.judge_reason


# 10. Test: Tier and model blinding in prompt
def test_judge_tier_and_model_blinding() -> None:
    """Prompt sent to judge must not contain candidate tier names, candidate model IDs, or costs."""
    _run_id, step_id = _create_sample_run_with_step(kind="text", content="Secret prompt reference")
    replay = _create_sample_replay(
        step_id,
        tier="nano",
        content="Candidate text without leaked identifiers.",
    )

    with Session(db_mod.engine) as session:
        step = session.get(Step, step_id)
        assert step is not None

        prompts = _build_judge_prompt(step, replay)
        full_prompt_text = " ".join(p["content"] for p in prompts)

        # Candidate tier name "nano" and model ID must NOT be leaked to judge
        assert "nano" not in full_prompt_text.lower()
        assert replay.model.lower() not in full_prompt_text.lower()
        # Cost and prices must not be leaked
        assert "cost_usd" not in full_prompt_text
        assert "0.00005" not in full_prompt_text
        assert "$" not in full_prompt_text


# 11. Test: Preservation of objective fields
def test_judge_preserves_objective_fields() -> None:
    """Judging an existing Evaluation updates judge fields while leaving objective_pass and detail intact."""
    _run_id, step_id = _create_sample_run_with_step(kind="text", content="Baseline reference")
    replay = _create_sample_replay(step_id, content="Candidate text")

    with Session(db_mod.engine) as session:
        step = session.get(Step, step_id)
        assert step is not None

        # 1. First run objective evaluation
        ev_initial = evaluate_replay(step, replay, session)
        orig_obj_pass = ev_initial.objective_pass
        orig_obj_detail = ev_initial.objective_detail
        assert orig_obj_pass is True
        assert orig_obj_detail is not None

        # 2. Run judge
        client = MockJudgeClient(default_score=0.88, default_verdict="pass")
        ev_judged = judge_replay(step, replay, client=client, session=session)

        # 3. Assert objective fields are preserved
        assert ev_judged.objective_pass == orig_obj_pass
        assert ev_judged.objective_detail == orig_obj_detail
        assert ev_judged.judge_score == 0.88


# 12. Test: Idempotency of judge_run
def test_judge_run_idempotency() -> None:
    """Running judge_run repeatedly does not create duplicate Evaluation rows."""
    run_id, step_id = _create_sample_run_with_step(kind="text")
    _create_sample_replay(step_id, tier="super")
    _create_sample_replay(step_id, tier="nano")

    client = MockJudgeClient(default_score=0.80, default_verdict="pass")

    # Run 1
    evals1 = judge_run(run_id=run_id, client=client)
    assert len(evals1) == 2

    with Session(db_mod.engine) as session:
        count_first = len(session.exec(select(Evaluation)).all())
        assert count_first == 2

    # Run 2 with force=False: should be idempotent and not create duplicate rows
    evals2 = judge_run(run_id=run_id, client=client, force=False)
    assert len(evals2) == 2

    with Session(db_mod.engine) as session:
        count_second = len(session.exec(select(Evaluation)).all())
        assert count_second == 2


# 13. Test: CLI judge command and refusal
def test_cli_judge_mock_and_real_refusal(capsys: pytest.CaptureFixture[str]) -> None:
    """Test CLI judge command in default mock mode and refusal when --real is passed."""
    run_id, step_id = _create_sample_run_with_step(kind="text")
    _create_sample_replay(step_id, tier="super")

    # 1. Refusal with --real
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["judge", run_id, "--real"])
    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "REFUSAL: Real API calls are disabled in Phase 2c." in err

    # 2. Successful mock judge via CLI
    cli.main(["judge", run_id])
    out = capsys.readouterr().out
    assert f"Downshift Replay Quality Report: Run {run_id}" in out
    assert "Tag" in out
    assert "Tier" in out
    assert "Pass Rate" in out
    assert "Avg Score" in out
    assert "Total Evaluations Judged" in out


def test_judge_replay_refuses_live_client_when_pricing_unverified() -> None:
    """judge_replay refuses execution if passed a live OpenAI client when live pricing is unverified."""
    from openai import OpenAI

    _run_id, step_id = _create_sample_run_with_step(kind="text")
    replay = _create_sample_replay(step_id, tier="super")

    with Session(db_mod.engine) as session:
        step = session.get(Step, step_id)
        assert step is not None

    live_client = OpenAI(api_key="mock-key", base_url="https://api.mock.nebius/v1")
    with pytest.raises(
        RuntimeError, match="REFUSAL: Cannot execute judge against live provider client"
    ):
        judge_replay(step=step, replay=replay, client=live_client)


def test_judge_run_and_replay_refuses_adversarial_client_types_when_pricing_unverified() -> None:
    """judge_run and judge_replay refuse execution for live, subclassed, or proxied clients."""
    from openai import OpenAI

    run_id, step_id = _create_sample_run_with_step(kind="text")
    replay = _create_sample_replay(step_id, tier="super")

    with Session(db_mod.engine) as session:
        step = session.get(Step, step_id)
        assert step is not None

    class CustomSubclassOpenAI(OpenAI):
        pass

    class ProxyClient:
        def __init__(self) -> None:
            self.client = OpenAI(api_key="mock", base_url="https://api.mock.nebius/v1")
            self.chat = self.client.chat

    sub_client = CustomSubclassOpenAI(api_key="mock", base_url="https://api.mock.nebius/v1")
    proxy_client = ProxyClient()

    # judge_run refuses live subclass client upfront
    with pytest.raises(
        RuntimeError, match="REFUSAL: Cannot execute judge against live provider client"
    ):
        judge_run(run_id=run_id, client=sub_client)

    # judge_run refuses proxy client upfront
    with pytest.raises(
        RuntimeError, match="REFUSAL: Cannot execute judge against live provider client"
    ):
        judge_run(run_id=run_id, client=proxy_client)

    # judge_replay refuses proxy client
    with pytest.raises(
        RuntimeError, match="REFUSAL: Cannot execute judge against live provider client"
    ):
        judge_replay(step=step, replay=replay, client=proxy_client)

    # judge_run refuses client with is_mock=True attribute
    tricky_client = OpenAI(api_key="mock", base_url="https://api.mock.nebius/v1")
    tricky_client.is_mock = True  # type: ignore[attr-defined]
    with pytest.raises(
        RuntimeError, match="REFUSAL: Cannot execute judge against live provider client"
    ):
        judge_run(run_id=run_id, client=tricky_client)

    # judge_replay refuses subclassed MockJudgeClient that wraps a live client
    class SneakyJudgeMock(MockJudgeClient):
        def __init__(self) -> None:
            super().__init__()
            self.live_backend = OpenAI(api_key="mock", base_url="https://api.mock.nebius/v1")

    sneaky_judge = SneakyJudgeMock()
    with pytest.raises(
        RuntimeError, match="REFUSAL: Cannot execute judge against live provider client"
    ):
        judge_replay(step=step, replay=replay, client=sneaky_judge)

    # judge_replay refuses a client named MockJudgeClient defined outside server.app.judge
    class ExternalMockJudgeClient:
        pass

    ExternalMockJudgeClient.__name__ = "MockJudgeClient"
    external_mock = ExternalMockJudgeClient()
    with pytest.raises(
        RuntimeError, match="REFUSAL: Cannot execute judge against live provider client"
    ):
        judge_replay(step=step, replay=replay, client=external_mock)

    # judge_replay refuses a client with spoofed __module__ and __name__
    class SpoofedFullIdentityClient:
        pass

    SpoofedFullIdentityClient.__name__ = "MockJudgeClient"
    SpoofedFullIdentityClient.__module__ = "server.app.judge"
    spoofed_full = SpoofedFullIdentityClient()
    with pytest.raises(
        RuntimeError, match="REFUSAL: Cannot execute judge against live provider client"
    ):
        judge_replay(step=step, replay=replay, client=spoofed_full)

    # judge_replay refuses an object spoofing unittest.mock attributes
    class SpoofedMockAttributeClient:
        def __init__(self) -> None:
            self._mock_self = True

    spoofed_mock = SpoofedMockAttributeClient()
    with pytest.raises(
        RuntimeError, match="REFUSAL: Cannot execute judge against live provider client"
    ):
        judge_replay(step=step, replay=replay, client=spoofed_mock)

    # judge_replay refuses slotted proxy hiding a live client
    class SlottedJudgeProxy:
        __slots__ = ("_live_inner",)

        def __init__(self) -> None:
            self._live_inner = OpenAI(api_key="mock", base_url="https://api.mock.nebius/v1")

    slotted_proxy = SlottedJudgeProxy()
    with pytest.raises(
        RuntimeError, match="REFUSAL: Cannot execute judge against live provider client"
    ):
        judge_replay(step=step, replay=replay, client=slotted_proxy)

    # Verify execution_mode gate refusal when execution_mode is offline (default)
    with pytest.raises(RuntimeError, match="execution_mode is 'offline'"):
        judge_replay(step=step, replay=replay, client=sub_client, execution_mode="offline")

    # Verify execution_mode="live" still refuses when live pricing is unverified
    with pytest.raises(
        RuntimeError,
        match="REFUSAL: Cannot execute judge against live provider client: prices.yaml has verified_for_live=false",
    ):
        judge_replay(step=step, replay=replay, client=sub_client, execution_mode="live")

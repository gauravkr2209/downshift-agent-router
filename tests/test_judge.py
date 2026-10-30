"""Tests for Phase 2c LLM-based replay quality judging, bias controls, and CLI."""

import json
from typing import Any

import pytest
from sqlmodel import Session, select

from server.app import cli
from server.app import db as db_mod
from server.app.checks import evaluate_replay
from server.app.config import settings
from server.app.db import create_db_and_tables
from server.app.judge import (
    MockJudgeClient,
    _build_judge_prompt,
    _MockJudgeChoice,
    _MockJudgeMessage,
    _MockJudgeResponse,
    _MockJudgeUsage,
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


# 13. Test: CLI judge command, credential checks, and mock mode
def test_cli_judge_mock_and_real_safeguards(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test CLI judge command in default mock mode and safeguard refusals when --real is passed."""
    run_id, step_id = _create_sample_run_with_step(kind="text")
    _create_sample_replay(step_id, tier="super")

    # 1. Refusal with missing/placeholder API key
    monkeypatch.setattr(settings, "nebius_api_key", "your_api_key_here")
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["judge", run_id, "--real"])
    assert exc_info.value.code == 1
    assert "REFUSAL: NEBIUS_API_KEY is not configured in .env." in capsys.readouterr().err

    # 2. Refusal with missing base URL
    monkeypatch.setattr(settings, "nebius_api_key", "valid-key-123")
    monkeypatch.setattr(settings, "nebius_base_url", None)
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["judge", run_id, "--real"])
    assert exc_info.value.code == 1
    assert "REFUSAL: NEBIUS_BASE_URL is not configured in .env." in capsys.readouterr().err

    # 3. Refusal with unverified pricing
    monkeypatch.setattr(settings, "nebius_base_url", "https://api.tokenfactory.nebius.com/v1")
    monkeypatch.setattr(
        db_mod,
        "check_live_pricing_verified",
        lambda: (False, "prices.yaml has verified_for_live=false (or unset/not boolean True)."),
    )
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["judge", run_id, "--real"])
    assert exc_info.value.code == 1
    assert "REFUSAL: prices.yaml has verified_for_live=false" in capsys.readouterr().err

    # 4. Successful client construction when credentials and pricing verified
    monkeypatch.setattr(db_mod, "check_live_pricing_verified", lambda: (True, "Pricing verified"))
    monkeypatch.setattr(db_mod, "sync_prices_from_yaml", lambda: None)

    created_clients: list[Any] = []

    class MockOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            created_clients.append(self)
            self.chat = self
            self.completions = self

        def create(self, **kwargs: Any) -> Any:
            resp_payload = {
                "score": 0.90,
                "verdict": "pass",
                "reason": "Accurate candidate output.",
            }
            return _MockJudgeResponse(
                choices=[
                    _MockJudgeChoice(message=_MockJudgeMessage(content=json.dumps(resp_payload)))
                ]
            )

    import openai

    monkeypatch.setattr(openai, "OpenAI", MockOpenAI)

    cli.main(["judge", run_id, "--real"])
    out = capsys.readouterr().out
    assert "(Mode: LIVE - Nebius Token Factory Judge)" in out
    assert len(created_clients) == 1
    assert created_clients[0].kwargs["api_key"] == "valid-key-123"
    assert created_clients[0].kwargs["base_url"] == "https://api.tokenfactory.nebius.com/v1"
    assert created_clients[0].kwargs["max_retries"] == 0

    # 5. Default mock judge via CLI
    cli.main(["judge", run_id])
    out = capsys.readouterr().out
    assert f"Downshift Replay Quality Report: Run {run_id}" in out
    assert "Tag" in out
    assert "Tier" in out
    assert "Pass Rate" in out
    assert "Avg Score" in out
    assert "Total Evaluations Judged" in out


# 14. Test: Anti-length bias and concise accurate answers
def test_judge_anti_length_bias_evaluates_concise_response() -> None:
    """Concise responses capturing essential facts receive high scores without length penalty."""
    _run_id, step_id = _create_sample_run_with_step(
        kind="text",
        content=(
            "The architecture of NVIDIA Nemotron 3.5 Super VL utilizes 512 routed "
            "experts with 120B total parameters and 12B active parameters. It includes "
            "a C-RADIO vision encoder and supports up to 1M context tokens."
        ),
    )
    replay = _create_sample_replay(
        step_id,
        tier="super",
        content=(
            "Nemotron 3.5 Super VL: 120B total / 12B active parameters, 512 MoE experts, "
            "C-RADIO encoder, 1M context."
        ),
    )

    client = MockJudgeClient(
        default_score=0.92,
        default_verdict="pass",
        default_reason="Concise answer directly preserves all key technical facts without fluff.",
    )

    eval_rec = judge_replay(step=session_step(step_id), replay=replay, client=client)
    assert eval_rec.judge_score == 0.92
    assert "[PASS]" in str(eval_rec.judge_reason)
    assert "without fluff" in str(eval_rec.judge_reason)


# 15. Test: Hallucination detection penalization
def test_judge_penalizes_hallucinations() -> None:
    """Candidates introducing fabricated entities or claims are penalized with fail verdict."""
    _run_id, step_id = _create_sample_run_with_step(
        kind="text",
        content="Nebius Token Factory hosts three models: Nemotron 3.5 Super VL.",
    )
    replay = _create_sample_replay(
        step_id,
        tier="nano",
        content="Nebius Token Factory hosts GPT-5 Turbo and Claude 4 Opus with 2M tokens.",
    )

    client = MockJudgeClient(
        scenario_handlers={
            "GPT-5": {
                "score": 0.10,
                "verdict": "fail",
                "reason": "Critical hallucination: candidate fabricated unhosted third-party models.",
            }
        }
    )

    eval_rec = judge_replay(step=session_step(step_id), replay=replay, client=client)
    assert eval_rec.judge_score == 0.10
    assert "[FAIL]" in str(eval_rec.judge_reason)
    assert "Critical hallucination" in str(eval_rec.judge_reason)


# 16. Test: Critical factual omission penalization
def test_judge_penalizes_critical_factual_omissions() -> None:
    """Candidates omitting vital conclusions or required parameters receive fail verdict."""
    _run_id, step_id = _create_sample_run_with_step(
        kind="text",
        content="The model has 120B total parameters and 12B active parameters with 512 experts.",
    )
    replay = _create_sample_replay(
        step_id,
        tier="nano",
        content="The model is an AI model from NVIDIA.",
    )

    client = MockJudgeClient(
        scenario_handlers={
            "AI model from NVIDIA": {
                "score": 0.35,
                "verdict": "fail",
                "reason": "Major factual omission: failed to report parameter counts and expert specifications.",
            }
        }
    )

    eval_rec = judge_replay(step=session_step(step_id), replay=replay, client=client)
    assert eval_rec.judge_score == 0.35
    assert "[FAIL]" in str(eval_rec.judge_reason)
    assert "Major factual omission" in str(eval_rec.judge_reason)


# 17. Test: Task and step context preserved in judge prompt
def test_judge_prompt_includes_task_and_step_context() -> None:
    """Judge prompt includes both overall task prompt and immediate step instruction."""
    with Session(db_mod.engine) as session:
        run = Run(
            task_name="context_test",
            agent_name="agent_test",
            baseline_model="nvidia/Nemotron-3-Ultra-550b-a55b",
            status="completed",
        )
        session.add(run)
        session.commit()
        session.refresh(run)

        step = Step(
            run_id=run.id,
            idx=0,
            tag="final_answer",
            kind="text",
            model="nvidia/Nemotron-3-Ultra-550b-a55b",
            request={
                "messages": [
                    {"role": "system", "content": "You are a research agent."},
                    {
                        "role": "user",
                        "content": "Overall research question: What models are on Nebius?",
                    },
                    {"role": "assistant", "content": "Interim search queries conducted."},
                    {
                        "role": "user",
                        "content": "Synthesize the final answer based on all findings.",
                    },
                ]
            },
            response={"choices": [{"message": {"content": "Final baseline report."}}]},
        )
        session.add(step)
        session.commit()
        session.refresh(step)

        replay = Replay(
            step_id=step.id,
            tier="super",
            model="nvidia/nemotron-3-super-120b-a12b",
            mode="step_level",
            response={"choices": [{"message": {"content": "Final candidate report."}}]},
        )
        session.add(replay)
        session.commit()
        session.refresh(replay)

        prompt = _build_judge_prompt(step, replay)

    user_msg = prompt[1]["content"]

    assert "Overall research question: What models are on Nebius?" in user_msg
    assert "Synthesize the final answer based on all findings." in user_msg
    assert "Anti-Length Bias Rule" in prompt[0]["content"]
    assert "Factual Consistency & Hallucination Detection" in prompt[0]["content"]


# 18. Test: Do not silently overwrite existing evaluation results without force
def test_judge_preserves_existing_evaluations_without_force() -> None:
    """Judge does not silently overwrite existing evaluation results unless force=True."""
    run_id, step_id = _create_sample_run_with_step(kind="text")
    _create_sample_replay(step_id, tier="super")

    client_initial = MockJudgeClient(
        default_score=0.88, default_verdict="pass", default_reason="Original score."
    )
    evals1 = judge_run(run_id=run_id, client=client_initial, force=False)
    assert len(evals1) == 1
    assert evals1[0].judge_score == 0.88
    assert "Original score." in str(evals1[0].judge_reason)

    # Second invocation with a different client score but force=False: must keep original
    client_overwrite = MockJudgeClient(
        default_score=0.20, default_verdict="fail", default_reason="New failing score."
    )
    evals2 = judge_run(run_id=run_id, client=client_overwrite, force=False)
    assert len(evals2) == 1
    assert evals2[0].judge_score == 0.88
    assert "Original score." in str(evals2[0].judge_reason)

    # Third invocation with force=True: updates evaluation
    evals3 = judge_run(run_id=run_id, client=client_overwrite, force=True)
    assert len(evals3) == 1
    assert evals3[0].judge_score == 0.20
    assert "New failing score." in str(evals3[0].judge_reason)


# Helper for session step loading in tests
def session_step(step_id: str) -> Step:
    with Session(db_mod.engine) as session:
        step = session.get(Step, step_id)
        assert step is not None
        return step


# 19. Test: Ambiguous failure halts subsequent calls
def test_judge_live_ambiguous_failure_halts_subsequent_calls() -> None:
    """Live judge request encountering ambiguous failure (e.g. timeout) tags unreconciled cost and halts subsequent calls."""
    run_id, step1_id = _create_sample_run_with_step(kind="text", tag="step_1")
    with Session(db_mod.engine) as session:
        step2 = Step(
            run_id=run_id,
            idx=1,
            tag="step_2",
            kind="text",
            model="nvidia/Nemotron-3-Ultra-550b-a55b",
            response={"choices": [{"message": {"content": "Step 2 baseline."}}]},
        )
        session.add(step2)
        session.commit()
        session.refresh(step2)
        step2_id = step2.id

    rep1 = _create_sample_replay(step1_id, tier="super", content="Candidate 1")
    rep2 = _create_sample_replay(step2_id, tier="super", content="Candidate 2")

    class FailingJudgeClient:
        class _Chat:
            def __init__(self, client: "FailingJudgeClient") -> None:
                self.client = client
                self.completions = self._Completions(client)

            class _Completions:
                def __init__(self, client: "FailingJudgeClient") -> None:
                    self.client = client

                def create(self, *args: Any, **kwargs: Any) -> Any:
                    self.client.call_count += 1
                    if self.client.call_count == 1:
                        raise TimeoutError("Nebius connection timed out during live judge call")
                    return _MockJudgeResponse(
                        choices=[
                            _MockJudgeChoice(
                                message=_MockJudgeMessage(
                                    content=json.dumps(
                                        {"score": 0.9, "verdict": "pass", "reason": "ok"}
                                    )
                                )
                            )
                        ],
                        usage=_MockJudgeUsage(),
                    )

        def __init__(self) -> None:
            self.call_count = 0
            self.chat = self._Chat(self)

    client = FailingJudgeClient()

    with pytest.raises(
        RuntimeError,
        match="Halting live judging for run .* previous judge call encountered unreconciled judge costs",
    ):
        judge_run(run_id=run_id, client=client, is_live=True)

    # Exactly 1 request was dispatched; the second was blocked before dispatch
    assert client.call_count == 1

    with Session(db_mod.engine) as session:
        ev1 = session.exec(select(Evaluation).where(Evaluation.replay_id == rep1.id)).first()
        assert ev1 is not None
        assert ev1.judge_score is None
        assert ev1.judge_reason is not None
        assert "[ERROR]" in ev1.judge_reason
        assert "UNRECONCILED_JUDGE_COST" in ev1.judge_reason
        assert ev1.judge_tokens_in is None
        assert ev1.judge_cost_usd is None

        ev2 = session.exec(select(Evaluation).where(Evaluation.replay_id == rep2.id)).first()
        if ev2 is not None:
            assert ev2.judge_reason is None or "UNRECONCILED_JUDGE_COST" not in ev2.judge_reason


# 20. Test: A separate invocation is blocked if unreconciled judge costs exist
def test_judge_live_separate_invocation_is_blocked() -> None:
    """A subsequent invocation of judge_run refuses to run if the run has persistent UNRECONCILED_JUDGE_COST."""
    run_id, step_id = _create_sample_run_with_step(kind="text")
    rep = _create_sample_replay(step_id, tier="super")

    with Session(db_mod.engine) as session:
        ev = Evaluation(
            replay_id=rep.id,
            objective_pass=True,
            judge_score=None,
            judge_reason=(
                "[ERROR] TimeoutError: connection lost "
                "[UNRECONCILED_JUDGE_COST: live judge request dispatched but failed ambiguously; "
                "further live judging halted]"
            ),
            judge_model="nvidia/Nemotron-3-Ultra-550b-a55b",
        )
        session.add(ev)
        session.commit()

    called = False

    class DummyClient:
        class _Chat:
            class _Completions:
                def create(self, *args: Any, **kwargs: Any) -> Any:
                    nonlocal called
                    called = True
                    raise AssertionError("Should never be called")

            completions = _Completions()

        chat = _Chat()

    client = DummyClient()

    with pytest.raises(
        RuntimeError,
        match="Live judging refused for run .* run contains persistent unreconciled judge costs",
    ):
        judge_run(run_id=run_id, client=client, is_live=True)

    assert called is False


# 21. Test: --force cannot bypass unreconciled block
def test_judge_live_force_cannot_bypass_unreconciled_block() -> None:
    """--force flag cannot bypass the persistent UNRECONCILED_JUDGE_COST block."""
    run_id, step_id = _create_sample_run_with_step(kind="text")
    rep = _create_sample_replay(step_id, tier="super")

    with Session(db_mod.engine) as session:
        ev = Evaluation(
            replay_id=rep.id,
            objective_pass=True,
            judge_score=None,
            judge_reason=(
                "[ERROR] TimeoutError: connection dropped "
                "[UNRECONCILED_JUDGE_COST: live judge request dispatched but failed ambiguously; "
                "further live judging halted]"
            ),
            judge_model="nvidia/Nemotron-3-Ultra-550b-a55b",
        )
        session.add(ev)
        session.commit()

    called = False

    class DummyClient:
        class _Chat:
            class _Completions:
                def create(self, *args: Any, **kwargs: Any) -> Any:
                    nonlocal called
                    called = True
                    raise AssertionError("Should never be called")

            completions = _Completions()

        chat = _Chat()

    client = DummyClient()

    with pytest.raises(
        RuntimeError,
        match="Live judging refused for run .* run contains persistent unreconciled judge costs",
    ):
        judge_run(run_id=run_id, client=client, force=True, is_live=True)

    assert called is False


# 22. Test: Missing usage triggers persistent safety block
def test_judge_live_missing_usage_triggers_persistent_safety_block() -> None:
    """When a live judge call succeeds with HTTP 200 but returns no usage metadata, it triggers UNRECONCILED_JUDGE_COST."""
    run_id, step_id = _create_sample_run_with_step(kind="text")
    rep = _create_sample_replay(step_id, tier="super")

    # Client returns valid JSON choices but usage=None
    client = MockJudgeClient(
        default_score=0.9,
        default_verdict="pass",
        default_reason="Output looks good.",
        include_usage=False,
    )

    with Session(db_mod.engine) as session:
        step = session.get(Step, step_id)
        assert step is not None
        ev = judge_replay(step, rep, client=client, session=session, is_live=True)

    assert ev.judge_score is None
    assert ev.judge_reason is not None
    assert "[ERROR]" in ev.judge_reason
    assert "UNRECONCILED_JUDGE_COST" in ev.judge_reason
    assert "missing or malformed usage metadata" in ev.judge_reason
    assert ev.judge_tokens_in is None
    assert ev.judge_cost_usd is None

    # Next invocation for the run is blocked
    with pytest.raises(
        RuntimeError,
        match="Live judging refused for run .* run contains persistent unreconciled judge costs",
    ):
        judge_run(run_id=run_id, client=client, is_live=True)


# 23. Test: Successful calls reconcile token usage and cost
def test_judge_live_successful_calls_reconcile_token_usage_and_cost() -> None:
    """Successful live judge calls properly record judge_tokens_in, judge_tokens_out, and judge_cost_usd."""
    run_id, step_id = _create_sample_run_with_step(kind="text")
    rep = _create_sample_replay(step_id, tier="super")

    client = MockJudgeClient(
        default_score=0.88,
        default_verdict="pass",
        default_reason="Semantic accuracy preserved.",
        include_usage=True,
        prompt_tokens=250,
        completion_tokens=50,
    )

    evals = judge_run(run_id=run_id, client=client, is_live=True)
    assert len(evals) == 1
    ev = evals[0]

    assert ev.judge_score == 0.88
    assert ev.judge_reason is not None
    assert ev.judge_reason.startswith("[PASS]")
    assert ev.judge_tokens_in == 250
    assert ev.judge_tokens_out == 50
    assert ev.judge_cost_usd is not None
    # For Ultra ($1.00/Mtok in, $3.00/Mtok out):
    # 250 * 1e-6 + 50 * 3e-6 = 0.00025 + 0.00015 = 0.00040
    assert abs(ev.judge_cost_usd - 0.00040) < 1e-6

    # Verify replay record itself was not altered
    with Session(db_mod.engine) as session:
        rep_db = session.get(Replay, rep.id)
        assert rep_db is not None
        assert rep_db.cost_usd == rep.cost_usd


# 24. Test: Existing objective evaluations remain unchanged
def test_judge_live_preserves_existing_objective_evaluations() -> None:
    """LLM judging must never overwrite or modify pre-existing objective evaluation fields."""
    run_id, step_id = _create_sample_run_with_step(kind="text")
    rep = _create_sample_replay(step_id, tier="super")

    with Session(db_mod.engine) as session:
        step = session.get(Step, step_id)
        assert step is not None
        initial_ev = evaluate_replay(step, rep, session)
        obj_pass = initial_ev.objective_pass
        obj_detail = dict(initial_ev.objective_detail or {})
        initial_ev_id = initial_ev.id

    client = MockJudgeClient(
        default_score=0.95,
        default_verdict="pass",
        default_reason="High quality response.",
        include_usage=True,
        prompt_tokens=180,
        completion_tokens=40,
    )

    evals = judge_run(run_id=run_id, client=client, is_live=True)
    assert len(evals) == 1
    final_ev = evals[0]

    assert final_ev.id == initial_ev_id
    # Objective evaluation fields remain 100% intact
    assert final_ev.objective_pass == obj_pass
    assert final_ev.objective_detail == obj_detail
    # Semantic judge fields populated
    assert final_ev.judge_score == 0.95
    assert final_ev.judge_tokens_in == 180
    assert final_ev.judge_tokens_out == 40
    assert final_ev.judge_cost_usd is not None


# 25. Test: Cost ceiling aborts before dispatch
def test_judge_live_cost_ceiling_aborts_before_dispatch() -> None:
    """Judge run aborts before dispatch if prospective request cost exceeds cost ceiling."""
    run_id, step_id = _create_sample_run_with_step(kind="text")
    _create_sample_replay(step_id, tier="super")

    client_called = False

    class DummyClient:
        class _Chat:
            class _Completions:
                def create(self, *args: Any, **kwargs: Any) -> Any:
                    nonlocal client_called
                    client_called = True
                    raise AssertionError("Should never be called")

            completions = _Completions()

        chat = _Chat()

    client = DummyClient()

    # Budget ceiling set to an impossibly low $0.000001, which cannot cover prospective judge call
    with pytest.raises(
        RuntimeError,
        match="Judge cost ceiling of .* reached: .* exceeds budget",
    ):
        judge_run(run_id=run_id, client=client, is_live=True, cost_ceiling_usd=0.000001)

    assert client_called is False


# 26. Test: Truncated output explicitly handled
def test_judge_truncated_output_explicitly_handled() -> None:
    """Truncated judge response cutting off mid-JSON is explicitly flagged as truncated error."""
    _run_id, step_id = _create_sample_run_with_step(kind="text")
    replay = _create_sample_replay(step_id)

    with Session(db_mod.engine) as session:
        step = session.get(Step, step_id)
        assert step is not None

        truncated_raw = '{\n  "score": 0.95,\n  "verdict": "pass",\n  "reason": "The candidate output accurately'
        client = MockJudgeClient(scenario_handlers={"Compare the following": truncated_raw})
        ev = judge_replay(step, replay, client=client, session=session)

        assert ev.judge_score is None
        assert ev.judge_reason is not None
        assert ev.judge_reason.startswith("[ERROR]")
        assert "Truncated judge output" in ev.judge_reason


# 27. Test: Preamble and postamble around JSON are stripped cleanly
def test_judge_extracts_json_with_surrounding_preamble() -> None:
    """Judge extracts valid JSON even when conversational reasoning preambles are present."""
    _run_id, step_id = _create_sample_run_with_step(kind="text")
    replay = _create_sample_replay(step_id)

    with Session(db_mod.engine) as session:
        step = session.get(Step, step_id)
        assert step is not None

        raw_with_preamble = (
            "We are evaluating the candidate output against the reference output.\n"
            '{"score": 0.88, "verdict": "pass", "reason": "Candidate output preserves intent and accuracy."}\n'
            "End of evaluation."
        )
        client = MockJudgeClient(scenario_handlers={"Compare the following": raw_with_preamble})
        ev = judge_replay(step, replay, client=client, session=session)

        assert ev.judge_score == 0.88
        assert ev.judge_reason is not None
        assert ev.judge_reason.startswith("[PASS]")


# 28. Test: Missing numeric score for pass/fail is rejected
def test_judge_missing_score_for_pass_fail_is_rejected() -> None:
    """A pass/fail verdict without a numeric score must be marked as an error, never passing."""
    _run_id, step_id = _create_sample_run_with_step(kind="text")
    replay = _create_sample_replay(step_id)

    with Session(db_mod.engine) as session:
        step = session.get(Step, step_id)
        assert step is not None

        payload = {"score": None, "verdict": "pass", "reason": "Good output without numeric score."}
        client = MockJudgeClient(scenario_handlers={"Compare the following": json.dumps(payload)})
        ev = judge_replay(step, replay, client=client, session=session)

        assert ev.judge_score is None
        assert ev.judge_reason is not None
        assert ev.judge_reason.startswith("[ERROR]")
        assert "without a numeric score" in ev.judge_reason


# 29. Test: Empty explanation reason is rejected
def test_judge_empty_reason_is_rejected() -> None:
    """An empty or whitespace-only explanation reason must be marked as an error."""
    _run_id, step_id = _create_sample_run_with_step(kind="text")
    replay = _create_sample_replay(step_id)

    with Session(db_mod.engine) as session:
        step = session.get(Step, step_id)
        assert step is not None

        payload = {"score": 0.9, "verdict": "pass", "reason": "   "}
        client = MockJudgeClient(scenario_handlers={"Compare the following": json.dumps(payload)})
        ev = judge_replay(step, replay, client=client, session=session)

        assert ev.judge_score is None
        assert ev.judge_reason is not None
        assert ev.judge_reason.startswith("[ERROR]")
        assert "empty explanation reason" in ev.judge_reason

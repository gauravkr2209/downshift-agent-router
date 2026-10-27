"""LLM-based replay quality judging for assessing semantic equivalence."""

import json
import re
from typing import Any

from pydantic import BaseModel, Field
from sqlmodel import Session, col, select

from server.app import db as db_mod
from server.app.checks import evaluate_replay
from server.app.db import create_db_and_tables
from server.app.models import Evaluation, Replay, Run, Step
from server.app.replay import _execute_with_retry, _is_live_client, resolve_tier_model


class JudgeOutputSchema(BaseModel):
    """Structured output expected from LLM quality judge."""

    score: float | None = Field(default=None, ge=0.0, le=1.0)
    verdict: str  # "pass", "fail", "uncertain"
    reason: str


def _clean_markdown_json(text: str) -> str:
    """Strip markdown code fence wrappers from LLM response text."""
    text_clean = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text_clean, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text_clean


def _objective_checks_have_run(evaluation: Evaluation | None) -> bool:
    """Determine whether objective checks have actually been executed.

    In the Evaluation schema, objective_pass defaults to False.
    An existing Evaluation row cannot be treated as a genuine objective failure
    merely because objective_pass is False; objective checks have only truly run
    if objective_detail is a dictionary containing a non-empty list of check results
    under the 'checks' key.
    """
    if evaluation is None:
        return False
    detail = evaluation.objective_detail
    if not isinstance(detail, dict):
        return False
    checks = detail.get("checks")
    return isinstance(checks, list) and len(checks) > 0


def _parse_judge_response(raw_text: str) -> tuple[float | None, str, str]:
    """Parse raw LLM judge response into (score, verdict, reason).

    Returns:
        (score, verdict, reason) where verdict is one of 'pass', 'fail', 'uncertain', 'error'.
    """
    clean_text = _clean_markdown_json(raw_text)
    try:
        data = json.loads(clean_text)
        if not isinstance(data, dict):
            return (
                None,
                "error",
                f"Judge response parsed JSON is not an object: {clean_text[:120]}",
            )

        parsed = JudgeOutputSchema(**data)
        norm_verdict = parsed.verdict.lower().strip()
        if norm_verdict not in ("pass", "fail", "uncertain"):
            return (
                None,
                "error",
                f"Invalid verdict '{parsed.verdict}' returned by judge; expected pass, fail, or uncertain.",
            )

        # Enforce score=None for uncertain verdicts
        final_score = None if norm_verdict == "uncertain" else parsed.score
        return final_score, norm_verdict, parsed.reason
    except Exception as err:  # noqa: BLE001
        return (
            None,
            "error",
            f"Malformed judge output failed to parse: {err}. Raw text: {raw_text[:120]}",
        )


def _build_judge_prompt(step: Step, replay: Replay) -> list[dict[str, str]]:
    """Build tier-blinded, model-blinded prompt comparing Candidate Output to Reference Output."""
    system_prompt = (
        "You are an impartial, expert AI benchmark judge. Your role is to assess whether "
        "a 'Candidate Output' preserves the semantic intent, factual accuracy, and task "
        "utility of the 'Reference Output' for an autonomous agent step.\n\n"
        "Guidelines:\n"
        "- Do NOT penalize candidate responses simply because wording differs from reference.\n"
        "- For search tool calls: assess whether the candidate's query preserves the research intent "
        "and is appropriate to gather the needed information. Legitimate variations in phrasing, "
        "synonyms, or search terms are expected and acceptable.\n"
        "- For text steps: assess factual consistency, relevance, and completeness. Penalize hallucinations, "
        "omissions of vital conclusions, or unsupported claims.\n"
        "- If the available context is insufficient to determine correctness with confidence, do NOT guess. "
        "Return verdict 'uncertain' with score null.\n\n"
        "Output format:\n"
        "You must respond with valid JSON containing:\n"
        "{\n"
        '  "score": <float between 0.0 and 1.0, or null if uncertain>,\n'
        '  "verdict": "<pass | fail | uncertain>",\n'
        '  "reason": "<concise explanation grounded in the inputs>"\n'
        "}\n"
        "Scores >= 0.70 qualify for 'pass'. Scores < 0.70 qualify for 'fail'."
    )

    # Blinded context extraction: never include candidate tier, model id, price, or cost!
    orig_tools = (
        [tc.get("function") for tc in (step.tool_calls or [])] if step.kind == "tool_call" else None
    )
    rep_tools = (
        [tc.get("function") for tc in (replay.tool_calls or [])]
        if step.kind == "tool_call"
        else None
    )

    orig_content = (
        step.response.get("choices", [{}])[0].get("message", {}).get("content")
        if step.response and isinstance(step.response, dict)
        else None
    )
    rep_content = (
        replay.response.get("choices", [{}])[0].get("message", {}).get("content")
        if replay.response and isinstance(replay.response, dict)
        else None
    )

    user_payload: dict[str, Any] = {
        "step_tag": step.tag,
        "step_kind": step.kind,
        "reference_output": {
            "content": orig_content,
            "tool_calls": orig_tools,
        },
        "candidate_output": {
            "content": rep_content,
            "tool_calls": rep_tools,
        },
    }

    # Add historical query context from step.request messages if present
    messages = step.request.get("messages", []) if isinstance(step.request, dict) else []
    if messages:
        # Include the last user or system instruction for context
        user_payload["step_context"] = [
            {"role": m.get("role"), "content": m.get("content")}
            for m in messages[-2:]
            if isinstance(m, dict) and m.get("content")
        ]

    user_prompt = (
        f"Compare the following Candidate Output against the Reference Output for step '{step.tag}'.\n\n"
        f"{json.dumps(user_payload, indent=2)}\n\n"
        "Evaluate the quality and intent preservation. Return your judgment in the required JSON format."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


class _MockJudgeMessage(BaseModel):
    role: str = "assistant"
    content: str


class _MockJudgeChoice(BaseModel):
    index: int = 0
    message: _MockJudgeMessage
    finish_reason: str = "stop"


class _MockJudgeResponse(BaseModel):
    id: str = "judge-mock-cmpl"
    model: str = "nvidia/Nemotron-3-Ultra-550b-a55b"
    choices: list[_MockJudgeChoice] = Field(default_factory=list)


class MockJudgeClient:
    """Deterministic OpenAI-compatible mock client for replay quality judging tests.

    Note: Mock evaluation patterns here are test scaffolding to verify parsing,
    failure handling, and grading flow, not real benchmark results.
    """

    def __init__(
        self,
        default_score: float | None = 0.85,
        default_verdict: str = "pass",
        default_reason: str = "Candidate output preserves intent and accuracy.",
        scenario_handlers: dict[str, Any] | None = None,
    ) -> None:
        self.default_score = default_score
        self.default_verdict = default_verdict
        self.default_reason = default_reason
        self.scenario_handlers = scenario_handlers or {}
        self.chat = self._Chat(self)

    class _Chat:
        def __init__(self, outer: "MockJudgeClient") -> None:
            self.completions = self._Completions(outer)

        class _Completions:
            def __init__(self, outer: "MockJudgeClient") -> None:
                self.outer = outer

            def create(self, *args: Any, **kwargs: Any) -> _MockJudgeResponse:
                messages = kwargs.get("messages", [])
                user_content = ""
                for m in messages:
                    if isinstance(m, dict) and m.get("role") == "user":
                        user_content = str(m.get("content", ""))

                # Check custom scenario handlers first
                for pattern, handler in self.outer.scenario_handlers.items():
                    if pattern in user_content:
                        if callable(handler):
                            res = handler(*args, **kwargs)
                            if isinstance(res, _MockJudgeResponse):
                                return res
                            return _MockJudgeResponse(
                                choices=[
                                    _MockJudgeChoice(message=_MockJudgeMessage(content=str(res)))
                                ]
                            )
                        elif isinstance(handler, str):
                            return _MockJudgeResponse(
                                choices=[
                                    _MockJudgeChoice(message=_MockJudgeMessage(content=handler))
                                ]
                            )
                        elif isinstance(handler, Exception):
                            raise handler

                # Default fallback response
                resp_payload = {
                    "score": self.outer.default_score,
                    "verdict": self.outer.default_verdict,
                    "reason": self.outer.default_reason,
                }
                return _MockJudgeResponse(
                    choices=[
                        _MockJudgeChoice(
                            message=_MockJudgeMessage(content=json.dumps(resp_payload))
                        )
                    ]
                )


def judge_replay(
    step: Step,
    replay: Replay,
    client: Any = None,
    judge_model: str | None = None,
    session: Session | None = None,
    execution_mode: str = "offline",
) -> Evaluation:
    """Judge a single replay's semantic quality, updating the Evaluation row."""
    create_db_and_tables()
    if client is not None and _is_live_client(client):
        if execution_mode != "live":
            raise RuntimeError(
                f"REFUSAL: Cannot execute judge against live provider client: execution_mode is '{execution_mode}'. "
                "Explicitly set execution_mode='live' and verify pricing in prices.yaml to authorize live inference."
            )
        is_verified, reason = db_mod.check_live_pricing_verified()
        if not is_verified:
            raise RuntimeError(
                f"REFUSAL: Cannot execute judge against live provider client: {reason}"
            )

    close_session = False
    if session is None:
        session = Session(db_mod.engine)
        close_session = True

    try:
        # 1. Resolve judge model identifier
        if judge_model is None:
            _, resolved_id = resolve_tier_model("ultra", session)
            judge_model = resolved_id

        # 2. Ensure objective evaluation has been performed
        evaluation = session.exec(
            select(Evaluation).where(Evaluation.replay_id == replay.id)
        ).first()

        if evaluation is None or not _objective_checks_have_run(evaluation):
            # Objective checks have not actually been run; execute them now
            evaluation = evaluate_replay(step, replay, session)

        assert evaluation is not None

        # 3. For an objectively invalid replay, record a genuine objective failure without calling the LLM judge
        if not evaluation.objective_pass:
            evaluation.judge_score = 0.0
            evaluation.judge_reason = "[FAIL] Short-circuited: Replay failed objective checks."
            evaluation.judge_model = judge_model
            session.add(evaluation)
            session.commit()
            session.refresh(evaluation)
            return evaluation

        # 4. Replay passed objective checks: call LLM judge
        if client is None:
            client = MockJudgeClient()

        prompt_messages = _build_judge_prompt(step, replay)
        call_kwargs = {
            "model": judge_model,
            "messages": prompt_messages,
            "temperature": 0.0,
            "max_tokens": 500,
        }

        final_score: float | None = None
        final_verdict: str = "error"
        final_reason: str = ""

        try:
            raw_response = _execute_with_retry(
                client.chat.completions.create,
                min_wait=None,
                max_wait=None,
                multiplier=None,
                attempts=None,
                **call_kwargs,
            )
            choices = getattr(raw_response, "choices", [])
            if choices:
                msg = getattr(choices[0], "message", None)
                content = getattr(msg, "content", "") if msg else ""
                final_score, final_verdict, final_reason = _parse_judge_response(content)
            else:
                final_score = None
                final_verdict = "error"
                final_reason = "Judge returned response with empty choices."
        except Exception as err:  # noqa: BLE001
            final_score = None
            final_verdict = "error"
            final_reason = f"Judge execution threw exception: {err}"

        # 5. Format judge_reason with standardized verdict prefix
        if final_verdict == "pass":
            formatted_reason = f"[PASS] {final_reason}"
        elif final_verdict == "fail":
            formatted_reason = f"[FAIL] {final_reason}"
        elif final_verdict == "uncertain":
            # For uncertain result, judge_score must be None
            final_score = None
            formatted_reason = f"[UNCERTAIN] {final_reason}"
        else:
            # For judge failure, judge_score must be None
            final_score = None
            formatted_reason = f"[ERROR] {final_reason}"

        evaluation.judge_score = final_score
        evaluation.judge_reason = formatted_reason
        evaluation.judge_model = judge_model
        session.add(evaluation)
        session.commit()
        session.refresh(evaluation)
        return evaluation
    finally:
        if close_session:
            session.close()


def judge_run(
    run_id: str,
    client: Any = None,
    judge_model: str | None = None,
    force: bool = False,
    session: Session | None = None,
    execution_mode: str = "offline",
) -> list[Evaluation]:
    """Execute LLM quality judging for all replays of a run without creating duplicate rows."""
    create_db_and_tables()
    if client is not None and _is_live_client(client):
        if execution_mode != "live":
            raise RuntimeError(
                f"REFUSAL: Cannot execute judge against live provider client: execution_mode is '{execution_mode}'. "
                "Explicitly set execution_mode='live' and verify pricing in prices.yaml to authorize live inference."
            )
        is_verified, reason = db_mod.check_live_pricing_verified()
        if not is_verified:
            raise RuntimeError(
                f"REFUSAL: Cannot execute judge against live provider client: {reason}"
            )

    close_session = False
    if session is None:
        session = Session(db_mod.engine)
        close_session = True

    try:
        run_record = session.get(Run, run_id)
        if not run_record:
            raise ValueError(f"Run '{run_id}' not found in database.")

        if judge_model is None:
            _, resolved_id = resolve_tier_model("ultra", session)
            judge_model = resolved_id

        steps = session.exec(
            select(Step).where(Step.run_id == run_id).order_by(col(Step.idx))
        ).all()

        evaluations: list[Evaluation] = []
        for step in steps:
            replays = session.exec(
                select(Replay).where(Replay.step_id == step.id).order_by(col(Replay.tier))
            ).all()

            for replay in replays:
                if not force:
                    existing = session.exec(
                        select(Evaluation).where(Evaluation.replay_id == replay.id)
                    ).first()
                    # Skip if already judged with valid score or explicit verdict prefix AND objective checks actually ran
                    if (
                        existing
                        and _objective_checks_have_run(existing)
                        and existing.judge_reason
                        and existing.judge_model
                    ):
                        evaluations.append(existing)
                        continue

                eval_rec = judge_replay(
                    step=step,
                    replay=replay,
                    client=client,
                    judge_model=judge_model,
                    session=session,
                    execution_mode=execution_mode,
                )
                evaluations.append(eval_rec)

        return evaluations
    finally:
        if close_session:
            session.close()

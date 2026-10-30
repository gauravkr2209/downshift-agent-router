"""LLM-based replay quality judging for assessing semantic equivalence."""

import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from sqlmodel import Session, col, select

from server.app import db as db_mod
from server.app.checks import evaluate_replay
from server.app.db import calculate_cost, create_db_and_tables
from server.app.models import Evaluation, Replay, Run, Step
from server.app.replay import _execute_with_retry, estimate_input_tokens, resolve_tier_model


class JudgeOutputSchema(BaseModel):
    """Structured output expected from LLM quality judge."""

    model_config = ConfigDict(extra="ignore")

    score: float | None = Field(default=None, ge=0.0, le=1.0)
    verdict: str  # "pass", "fail", "uncertain"
    reason: str
    factual_consistency: float | None = Field(default=None, ge=0.0, le=1.0)
    completeness: float | None = Field(default=None, ge=0.0, le=1.0)
    hallucination_detected: bool | None = Field(default=None)


def _clean_markdown_json(text: str) -> str:
    """Strip markdown code fence wrappers and extract JSON object."""
    text_clean = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text_clean, re.IGNORECASE)
    if match:
        text_clean = match.group(1).strip()

    # If text doesn't start with '{', extract outermost '{ ... }'
    if not text_clean.startswith("{"):
        brace_start = text_clean.find("{")
        brace_end = text_clean.rfind("}")
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            text_clean = text_clean[brace_start : brace_end + 1].strip()
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

        if not parsed.reason or not parsed.reason.strip():
            return (
                None,
                "error",
                "Judge returned empty explanation reason.",
            )

        if norm_verdict in ("pass", "fail") and parsed.score is None:
            return (
                None,
                "error",
                f"Judge returned verdict '{norm_verdict}' without a numeric score.",
            )

        # Enforce score=None for uncertain verdicts
        final_score = None if norm_verdict == "uncertain" else parsed.score
        return final_score, norm_verdict, parsed.reason.strip()
    except Exception as err:  # noqa: BLE001
        is_truncated = (
            raw_text.count("{") > raw_text.count("}")
            or (clean_text.startswith("{") and not clean_text.endswith("}"))
            or "Unterminated string" in str(err)
        )
        if is_truncated:
            return (
                None,
                "error",
                f"Truncated judge output failed to complete JSON object: {err}. Raw text: {raw_text[:120]}",
            )
        return (
            None,
            "error",
            f"Malformed judge output failed to parse: {err}. Raw text: {raw_text[:120]}",
        )


def _build_judge_prompt(step: Step, replay: Replay) -> list[dict[str, str]]:
    """Build tier-blinded, model-blinded prompt comparing Candidate Output to Reference Output."""
    system_prompt = (
        "You are an impartial, expert AI benchmark judge evaluating autonomous agent execution steps.\n"
        "Your role is to assess whether a 'Candidate Output' preserves the semantic intent, factual accuracy, "
        "and task utility of the step given the original task context and source evidence.\n\n"
        "Ground Truth & Reference Grounding Guidelines:\n"
        "- The 'Reference Output' was generated by an autonomous baseline run; treat it as an informative benchmark, "
        "NOT as unquestionable or infallible ground truth.\n"
        "- If the source evidence or task context proves that the candidate output is factually accurate, do NOT "
        "penalize it simply because it diverges from an incomplete or imperfect reference.\n"
        "- Distinguish factual correctness from reference similarity: Do NOT measure word overlap, sentence structure, "
        "or stylistic mimicry. Valid alternative phrasing, synonyms, and restructured presentations are expected and acceptable.\n\n"
        "Evaluation Dimensions:\n"
        "1. Factual Consistency & Hallucination Detection: The candidate must not introduce unsupported, fabricated entities, "
        "metrics, or claims. Factual claims must be verifiable against the task context or source text. Heavily penalize hallucinations.\n"
        "2. Completeness & Omissions: The candidate must capture key findings, conclusions, and entities. "
        "Penalize omissions of critical details required by the task context.\n"
        "3. Instruction Adherence & Relevance: The candidate must address the prompt directly and adhere to requested "
        "formatting (e.g. search syntax, summaries, structured reports).\n"
        "4. Output Quality: Clear, coherent, professional presentation.\n\n"
        "Anti-Length Bias Rule:\n"
        "- Do NOT reward candidate answers simply for being longer or containing more words.\n"
        "- A concise, direct candidate response that captures all essential facts is equal or superior to a verbose one.\n"
        "- Do NOT penalize brevity if factual completeness is maintained.\n\n"
        "Step-Specific Guidance:\n"
        "- For search tool calls or query formulation ('search_query'): Evaluate whether the candidate query preserves "
        "the research intent and will retrieve the required information. Synonyms, operator variations, and equivalent "
        "keywords are completely acceptable.\n"
        "- For document summaries ('summarize_page'): Evaluate whether key technical facts (model names, specifications, "
        "architecture details) from the source document provided in context are accurately preserved.\n"
        "- For final answers ('final_answer'): Evaluate whether all parts of the user research question are accurately "
        "and factually addressed without confabulation.\n\n"
        "Uncertainty Rule:\n"
        "- If the provided context is insufficient to determine correctness with confidence, do NOT guess. "
        "Return verdict 'uncertain' with score null.\n\n"
        "Strict JSON Output Requirements:\n"
        "- You MUST output ONLY a valid, compact JSON object.\n"
        "- Do NOT output any reasoning preamble, thoughts, introductory text, or markdown code fences outside the JSON.\n"
        "- Begin directly with '{' and end with '}'.\n\n"
        "Output Format:\n"
        "{\n"
        '  "score": <float between 0.0 and 1.0, or null if uncertain>,\n'
        '  "verdict": "<pass | fail | uncertain>",\n'
        '  "reason": "<concise explanation addressing factual consistency, completeness, and quality>"\n'
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

    # Add task and step context from step.request messages if present
    messages = step.request.get("messages", []) if isinstance(step.request, dict) else []
    if messages:
        first_user = next(
            (m.get("content") for m in messages if isinstance(m, dict) and m.get("role") == "user"),
            None,
        )
        immediate_instruction = (
            messages[-1].get("content") if isinstance(messages[-1], dict) else None
        )

        task_context: dict[str, Any] = {}
        if first_user:
            task_context["overall_task_prompt"] = first_user
        if immediate_instruction and (not first_user or immediate_instruction != first_user):
            task_context["immediate_step_instruction"] = immediate_instruction

        if task_context:
            user_payload["task_context"] = task_context

    user_prompt = (
        f"Compare the following Candidate Output against the Reference Output for step '{step.tag}'.\n\n"
        f"{json.dumps(user_payload, indent=2)}\n\n"
        "Evaluate the quality and intent preservation. Return your judgment as a valid, compact JSON object only with no preamble."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


class _MockJudgeUsage(BaseModel):
    prompt_tokens: int = 150
    completion_tokens: int = 50
    total_tokens: int = 200


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
    usage: _MockJudgeUsage | None = None


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
        include_usage: bool = True,
        prompt_tokens: int = 150,
        completion_tokens: int = 50,
    ) -> None:
        self.default_score = default_score
        self.default_verdict = default_verdict
        self.default_reason = default_reason
        self.scenario_handlers = scenario_handlers or {}
        self.include_usage = include_usage
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
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

                default_usage = (
                    _MockJudgeUsage(
                        prompt_tokens=self.outer.prompt_tokens,
                        completion_tokens=self.outer.completion_tokens,
                        total_tokens=self.outer.prompt_tokens + self.outer.completion_tokens,
                    )
                    if self.outer.include_usage
                    else None
                )

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
                                ],
                                usage=default_usage,
                            )
                        elif isinstance(handler, str):
                            return _MockJudgeResponse(
                                choices=[
                                    _MockJudgeChoice(message=_MockJudgeMessage(content=handler))
                                ],
                                usage=default_usage,
                            )
                        elif isinstance(handler, dict):
                            return _MockJudgeResponse(
                                choices=[
                                    _MockJudgeChoice(
                                        message=_MockJudgeMessage(content=json.dumps(handler))
                                    )
                                ],
                                usage=default_usage,
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
                    ],
                    usage=default_usage,
                )


def judge_replay(
    step: Step,
    replay: Replay,
    client: Any = None,
    judge_model: str | None = None,
    session: Session | None = None,
    is_live: bool = False,
) -> Evaluation:
    """Judge a single replay's semantic quality, updating the Evaluation row."""
    create_db_and_tables()
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
            evaluation.judge_tokens_in = None
            evaluation.judge_tokens_out = None
            evaluation.judge_cost_usd = None
            session.add(evaluation)
            session.commit()
            session.refresh(evaluation)
            return evaluation

        # 4. Replay passed objective checks: call LLM judge
        if client is None:
            client = MockJudgeClient()

        prompt_messages = _build_judge_prompt(step, replay)
        call_kwargs: dict[str, Any] = {
            "model": judge_model,
            "messages": prompt_messages,
            "temperature": 0.0,
            "max_tokens": 600,
            "response_format": {"type": "json_object"},
        }

        final_score: float | None = None
        final_verdict: str = "error"
        final_reason: str = ""
        tokens_in: int | None = None
        tokens_out: int | None = None
        judge_cost: float | None = None

        request_dispatched = False
        try:
            request_dispatched = True
            try:
                raw_response = _execute_with_retry(
                    client.chat.completions.create,
                    min_wait=None,
                    max_wait=None,
                    multiplier=None,
                    attempts=1,
                    **call_kwargs,
                )
            except TypeError as type_err:
                if "response_format" in str(type_err) and "response_format" in call_kwargs:
                    fallback_kwargs = dict(call_kwargs)
                    fallback_kwargs.pop("response_format", None)
                    raw_response = _execute_with_retry(
                        client.chat.completions.create,
                        min_wait=None,
                        max_wait=None,
                        multiplier=None,
                        attempts=1,
                        **fallback_kwargs,
                    )
                else:
                    raise

            # Extract usage metadata
            if isinstance(raw_response, dict):
                usage = raw_response.get("usage")
            else:
                usage = getattr(raw_response, "usage", None)

            prompt_tokens: Any = None
            completion_tokens: Any = None
            if usage is not None:
                if isinstance(usage, dict):
                    prompt_tokens = usage.get("prompt_tokens")
                    completion_tokens = usage.get("completion_tokens")
                else:
                    prompt_tokens = getattr(usage, "prompt_tokens", None)
                    completion_tokens = getattr(usage, "completion_tokens", None)

            if is_live and (usage is None or prompt_tokens is None or completion_tokens is None):
                final_score = None
                final_verdict = "error"
                final_reason = (
                    "Response has missing or malformed usage metadata "
                    "[UNRECONCILED_JUDGE_COST: live judge request completed without verifiable usage metadata; further live judging halted]"
                )
                tokens_in = None
                tokens_out = None
                judge_cost = None
            else:
                if prompt_tokens is not None and completion_tokens is not None:
                    tokens_in = int(prompt_tokens)
                    tokens_out = int(completion_tokens)
                    try:
                        judge_cost = calculate_cost(judge_model, tokens_in, tokens_out, session)
                    except Exception:  # noqa: BLE001
                        judge_cost = None

                choices = (
                    raw_response.get("choices", [])
                    if isinstance(raw_response, dict)
                    else getattr(raw_response, "choices", [])
                )
                if choices:
                    first_choice = choices[0]
                    msg = (
                        first_choice.get("message")
                        if isinstance(first_choice, dict)
                        else getattr(first_choice, "message", None)
                    )
                    content = (
                        msg.get("content", "")
                        if isinstance(msg, dict)
                        else getattr(msg, "content", "")
                        if msg
                        else ""
                    )
                    final_score, final_verdict, final_reason = _parse_judge_response(content)
                else:
                    final_score = None
                    final_verdict = "error"
                    final_reason = "Judge returned response with empty choices."
        except Exception as err:  # noqa: BLE001
            final_score = None
            final_verdict = "error"
            tokens_in = None
            tokens_out = None
            judge_cost = None
            if is_live and request_dispatched:
                final_reason = (
                    f"{type(err).__name__}: {err} [UNRECONCILED_JUDGE_COST: live judge request dispatched "
                    f"but failed ambiguously; further live judging halted]"
                )
            else:
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
        evaluation.judge_tokens_in = tokens_in
        evaluation.judge_tokens_out = tokens_out
        evaluation.judge_cost_usd = judge_cost
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
    is_live: bool = False,
    cost_ceiling_usd: float | None = None,
) -> list[Evaluation]:
    """Execute LLM quality judging for all replays of a run without creating duplicate rows."""
    create_db_and_tables()
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

        if is_live:
            # Persistent safety block: if any previous live attempt on this run encountered
            # ambiguous billing / unreconciled judge costs, refuse further live judging until manual review.
            unreconciled_existing = session.exec(
                select(Evaluation)
                .join(Replay, col(Replay.id) == col(Evaluation.replay_id))
                .join(Step, col(Step.id) == col(Replay.step_id))
                .where(
                    Step.run_id == run_id,
                    col(Evaluation.judge_reason).contains("UNRECONCILED_JUDGE_COST"),
                )
            ).all()
            if unreconciled_existing:
                affected = [f"eval {e.id} (replay {e.replay_id})" for e in unreconciled_existing]
                raise RuntimeError(
                    f"Live judging refused for run '{run_id}': run contains persistent unreconciled judge costs "
                    f"from previous ambiguous failure ({', '.join(affected)}). Manual review required before further live requests."
                )

        steps = session.exec(
            select(Step).where(Step.run_id == run_id).order_by(col(Step.idx))
        ).all()

        evaluations: list[Evaluation] = []
        has_unreconciled_judge_cost = False

        ceiling = cost_ceiling_usd if cost_ceiling_usd is not None else (0.07 if is_live else None)
        reserved_cost = 0.0
        cumulative_actual_cost = 0.0

        if ceiling is not None:
            existing_evals = session.exec(
                select(Evaluation)
                .join(Replay, col(Replay.id) == col(Evaluation.replay_id))
                .join(Step, col(Step.id) == col(Replay.step_id))
                .where(
                    Step.run_id == run_id,
                    col(Evaluation.judge_cost_usd).is_not(None),
                )
            ).all()
            cumulative_actual_cost = sum(e.judge_cost_usd or 0.0 for e in existing_evals)

        for step in steps:
            replays = session.exec(
                select(Replay).where(Replay.step_id == step.id).order_by(col(Replay.tier))
            ).all()

            for replay in replays:
                if is_live and has_unreconciled_judge_cost:
                    raise RuntimeError(
                        f"Halting live judging for run '{run_id}': previous judge call encountered unreconciled judge costs. "
                        "Subsequent live requests blocked to prevent untracked spend."
                    )

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

                # Prospective cost reservation before live dispatch
                est_req_cost = 0.0
                if ceiling is not None:
                    # Check if replay fails objective evaluation first to avoid reserving budget for short-circuited steps
                    existing_eval = session.exec(
                        select(Evaluation).where(Evaluation.replay_id == replay.id)
                    ).first()
                    if existing_eval is None or not _objective_checks_have_run(existing_eval):
                        existing_eval = evaluate_replay(step, replay, session)

                    if existing_eval.objective_pass:
                        prompt_msgs = _build_judge_prompt(step, replay)
                        call_kwargs = {
                            "model": judge_model,
                            "messages": prompt_msgs,
                            "max_tokens": 600,
                        }
                        est_in = estimate_input_tokens(call_kwargs)
                        est_req_cost = calculate_cost(judge_model, est_in, 600, session)
                        if (cumulative_actual_cost + reserved_cost + est_req_cost) > ceiling:
                            raise RuntimeError(
                                f"Judge cost ceiling of ${ceiling:.6f} reached: current spent+reserved "
                                f"${cumulative_actual_cost + reserved_cost:.6f} plus estimated request cost "
                                f"${est_req_cost:.6f} exceeds budget of ${ceiling:.6f}. Aborting live judge call before dispatch."
                            )
                        reserved_cost += est_req_cost

                eval_rec = judge_replay(
                    step=step,
                    replay=replay,
                    client=client,
                    judge_model=judge_model,
                    session=session,
                    is_live=is_live,
                )
                evaluations.append(eval_rec)
                if (
                    is_live
                    and eval_rec.judge_reason
                    and "UNRECONCILED_JUDGE_COST" in eval_rec.judge_reason
                ):
                    has_unreconciled_judge_cost = True
                else:
                    if ceiling is not None and est_req_cost > 0.0:
                        if eval_rec.judge_cost_usd is not None:
                            cumulative_actual_cost += eval_rec.judge_cost_usd
                        reserved_cost = max(0.0, reserved_cost - est_req_cost)

        return evaluations
    finally:
        if close_session:
            session.close()

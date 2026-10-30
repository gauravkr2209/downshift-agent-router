"""Replay engine for executing recorded agent steps across cheaper model tiers."""

import copy
import json
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import BaseModel, Field
from sqlmodel import Session, col, select
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from sdk.downshift.recorder import _serialize_payload
from server.app import db as db_mod
from server.app.config import settings
from server.app.db import (
    calculate_cost,
    create_db_and_tables,
    resolve_model_price,
    sync_prices_from_yaml,
)
from server.app.models import Evaluation, Replay, Run, Step


def _is_retryable_exception(exc: BaseException) -> bool:
    """Determine whether an exception represents a retryable 429/5xx error."""
    status_code = getattr(exc, "status_code", None)
    if status_code in (429, 500, 502, 503, 504):
        return True

    exc_name = type(exc).__name__.lower()
    target_names = ("ratelimit", "internalserver", "apiconnection", "timeout")
    return any(target in exc_name for target in target_names)


def _execute_with_retry(
    fn: Callable[..., Any],
    *args: Any,
    min_wait: float | None = None,
    max_wait: float | None = None,
    multiplier: float | None = None,
    attempts: int | None = None,
    **kwargs: Any,
) -> Any:
    """Execute a callable with tenacity retrying on 429/5xx errors."""
    min_w = min_wait if min_wait is not None else settings.replay_retry_min_wait
    max_w = max_wait if max_wait is not None else settings.replay_retry_max_wait
    mult = multiplier if multiplier is not None else settings.replay_retry_multiplier
    att = attempts if attempts is not None else settings.replay_retry_attempts

    retryer = Retrying(
        retry=retry_if_exception(_is_retryable_exception),
        stop=stop_after_attempt(att),
        wait=wait_exponential(multiplier=mult, min=min_w, max=max_w),
        reraise=True,
    )
    return retryer(fn, *args, **kwargs)


def resolve_tier_model(tier: str, session: Session) -> tuple[str, str]:
    """Resolve tier identifier or alias into (normalized_tier, real_model_id)."""
    create_db_and_tables()
    tier_clean = tier.lower().strip()

    if tier_clean in ("super", "nemotron_super"):
        normalized_tier = "super"
        alias = "nemotron_super"
        env_model = settings.super_model_id
    elif tier_clean in ("nano", "nemotron_nano"):
        normalized_tier = "nano"
        alias = "nemotron_nano"
        env_model = settings.nano_model_id
    elif tier_clean in ("ultra", "nemotron_ultra"):
        normalized_tier = "ultra"
        alias = "nemotron_ultra"
        env_model = settings.ultra_model_id
    else:
        normalized_tier = tier_clean
        alias = tier_clean
        env_model = None

    if env_model and not env_model.startswith("PLACEHOLDER"):
        real_id, _ = resolve_model_price(env_model, session)
        return normalized_tier, real_id

    real_id, _ = resolve_model_price(alias, session)
    return normalized_tier, real_id


def estimate_input_tokens(request: dict[str, Any]) -> int:
    """Conservatively estimate input tokens from serialized messages and tool definitions.

    Uses character heuristics (~4 characters per token plus protocol framing overhead).
    """
    messages = request.get("messages", [])
    approx_chars = 0
    if isinstance(messages, list):
        for m in messages:
            if isinstance(m, dict):
                c = m.get("content", "")
                if isinstance(c, str):
                    approx_chars += len(c)
                elif isinstance(c, list):
                    for item in c:
                        if isinstance(item, dict):
                            approx_chars += len(str(item.get("text", "")))
                        else:
                            approx_chars += len(str(item))
            elif isinstance(m, str):
                approx_chars += len(m)

    tools = request.get("tools")
    if isinstance(tools, (list, tuple)):
        for tool in tools:
            try:
                serialized_tool = _serialize_payload(tool)
                approx_chars += len(json.dumps(serialized_tool))
            except Exception:  # noqa: BLE001
                approx_chars += len(str(tool))
    elif tools is not None:
        try:
            serialized_tool = _serialize_payload(tools)
            approx_chars += len(json.dumps(serialized_tool))
        except Exception:  # noqa: BLE001
            approx_chars += len(str(tools))

    framing_tokens = 50 if (isinstance(messages, list) and messages) or tools else 0
    return max(1, (approx_chars // 4) + framing_tokens)


class _MockUsage(BaseModel):
    prompt_tokens: int | None = 150
    completion_tokens: int | None = 60
    total_tokens: int | None = 210


class _MockToolCallFunction(BaseModel):
    name: str
    arguments: str


class _MockToolCall(BaseModel):
    id: str = Field(default="call_mock_replay")
    type: str = "function"
    function: _MockToolCallFunction


class _MockMessage(BaseModel):
    role: str = "assistant"
    content: str | None = None
    tool_calls: list[_MockToolCall] | None = None
    reasoning_content: str | None = None
    thought: str | None = None


class _MockChoice(BaseModel):
    index: int = 0
    message: _MockMessage
    finish_reason: str = "stop"


class _MockResponse(BaseModel):
    id: str = "replay-mock-cmpl"
    model: str
    usage: _MockUsage | None = None
    choices: list[_MockChoice] = Field(default_factory=list)


class MockReplayClient:
    """Deterministic OpenAI-compatible mock client for per-tier step replay.

    Note: The hardcoded Nano tool call failure (returning invalid arguments)
    is test scaffolding designed to exercise objective check failure handling,
    not a benchmark result or reflection of actual model performance.
    """

    def __init__(
        self,
        tier_behaviors: dict[str, Any] | None = None,
        prompt_tokens: int = 150,
        completion_tokens: int = 60,
    ) -> None:
        self.tier_behaviors = tier_behaviors or {}
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.chat = self
        self.completions = self

    def create(self, *args: Any, **kwargs: Any) -> _MockResponse:
        """Simulate chat completion for a given tier model."""
        model = str(kwargs.get("model", "unknown"))
        model_lower = model.lower()

        # Match custom tier behavior by exact key or substring match
        matched_behavior = None
        for key, behavior in self.tier_behaviors.items():
            if key.lower() == model_lower or key.lower() in model_lower:
                matched_behavior = behavior
                break

        if isinstance(matched_behavior, BaseException):
            raise matched_behavior
        if callable(matched_behavior):
            return matched_behavior(kwargs)  # type: ignore[no-any-return]

        tools = kwargs.get("tools")
        tool_calls: list[_MockToolCall] | None = None
        finish_reason = "stop"
        content: str | None = None
        reasoning_content: str | None = None

        if isinstance(matched_behavior, dict):
            content = matched_behavior.get("content")
            reasoning_content = matched_behavior.get("reasoning_content")
            raw_tool_calls = matched_behavior.get("tool_calls")
            if raw_tool_calls:
                tool_calls = [
                    _MockToolCall(
                        id=tc.get("id", "call_mock"),
                        type="function",
                        function=_MockToolCallFunction(
                            name=tc["function"]["name"],
                            arguments=tc["function"]["arguments"]
                            if isinstance(tc["function"]["arguments"], str)
                            else json.dumps(tc["function"]["arguments"]),
                        ),
                    )
                    for tc in raw_tool_calls
                ]
                finish_reason = "tool_calls"
        elif tools:
            first_tool = tools[0]
            func_name = first_tool.get("function", {}).get("name", "tavily_search")
            if "nano" in model_lower:
                # Nano returns wrong argument for deterministic objective failure tests
                tool_args = json.dumps({"query": "wrong_nano_argument", "unrecognized_flag": True})
            else:
                # Super / Ultra return correct argument
                tool_args = json.dumps({"query": "NVIDIA Nemotron architecture specifications"})

            tool_calls = [
                _MockToolCall(
                    id=f"call_{model_lower[:10]}",
                    type="function",
                    function=_MockToolCallFunction(name=func_name, arguments=tool_args),
                )
            ]
            finish_reason = "tool_calls"
            reasoning_content = f"{model} reasoning: selected tool {func_name} for research step."
        else:
            if "nano" in model_lower:
                content = "Nano replay response: concise simulated step output."
                reasoning_content = "Nano reasoning trace."
            elif "super" in model_lower:
                content = "Super replay response: accurate simulated output with detailed findings."
                reasoning_content = "Super detailed step-by-step reasoning trace."
            else:
                content = f"{model} replay response: comprehensive synthesized output."
                reasoning_content = f"{model} deep reasoning trace."

        return _MockResponse(
            model=model,
            usage=_MockUsage(
                prompt_tokens=self.prompt_tokens,
                completion_tokens=self.completion_tokens,
                total_tokens=self.prompt_tokens + self.completion_tokens,
            ),
            choices=[
                _MockChoice(
                    message=_MockMessage(
                        role="assistant",
                        content=content,
                        tool_calls=tool_calls,
                        reasoning_content=reasoning_content,
                        thought=reasoning_content,
                    ),
                    finish_reason=finish_reason,
                )
            ],
        )


def replay_run(
    run_id: str,
    tiers: Sequence[str] | list[str],
    client: Any = None,
    max_workers: int = 4,
    max_steps: int | None = None,
    cost_ceiling_usd: float | None = None,
    force: bool = False,
    session: Session | None = None,
    retry_min_wait: float | None = None,
    retry_max_wait: float | None = None,
    retry_multiplier: float | None = None,
    retry_attempts: int | None = None,
    is_live: bool = False,
) -> list[Replay]:
    """Replay recorded run steps across candidate tiers without using the recorder."""
    create_db_and_tables()
    sync_prices_from_yaml()

    is_live = is_live or (client is not None and type(client).__name__ == "OpenAI")

    if client is None:
        if is_live:
            raise ValueError("A configured live client must be provided when is_live=True.")
        client = MockReplayClient()

    if is_live:
        max_workers = 1
        if retry_attempts is None:
            retry_attempts = 1

    close_session = False
    if session is None:
        session = Session(db_mod.engine)
        close_session = True

    try:
        run_record = session.get(Run, run_id)
        if not run_record:
            raise ValueError(f"Run '{run_id}' not found in database.")

        steps_query = select(Step).where(Step.run_id == run_id).order_by(col(Step.idx))
        steps = session.exec(steps_query).all()
        if max_steps is not None:
            steps = steps[:max_steps]

        if not steps:
            return []

        resolved_tiers = [resolve_tier_model(t, session) for t in tiers]

        if is_live:
            # Persistent safety block: if any previous live attempt on this run encountered
            # ambiguous billing / unreconciled costs, refuse further live replay until manual review.
            unreconciled_existing = session.exec(
                select(Replay)
                .join(Step, col(Step.id) == col(Replay.step_id))
                .where(
                    Step.run_id == run_id,
                    col(Replay.error).contains("UNRECONCILED_COST"),
                )
            ).all()
            if unreconciled_existing:
                affected = [f"step {r.step_id} tier {r.tier}" for r in unreconciled_existing]
                raise RuntimeError(
                    f"Live replay refused for run '{run_id}': run contains persistent unreconciled costs "
                    f"from previous ambiguous failure ({', '.join(affected)}). Manual review required before further live requests."
                )

        # Filter out steps already replayed for a tier unless force=True
        # Idempotent skip: only skip when existing replay has error IS NULL
        tasks: list[tuple[Step, str, str]] = []
        for step in steps:
            for norm_tier, model_id in resolved_tiers:
                if not force:
                    existing = session.exec(
                        select(Replay).where(
                            Replay.step_id == step.id,
                            Replay.tier == norm_tier,
                            col(Replay.error).is_(None),
                        )
                    ).first()
                    if existing:
                        continue
                tasks.append((step, norm_tier, model_id))
    finally:
        if close_session:
            session.close()

    if not tasks:
        return []

    cumulative_cost = 0.0
    cost_lock = threading.Lock()
    replays_created: list[Replay] = []
    replays_lock = threading.Lock()

    # Pre-call cost reservation and unreconciled usage tracking for live mode
    reserved_cost = 0.0
    cumulative_actual_cost = 0.0
    has_unreconciled_cost = False
    budget_lock = threading.Lock()

    def _execute_single(task: tuple[Step, str, str]) -> Replay:
        nonlocal cumulative_cost, reserved_cost, cumulative_actual_cost, has_unreconciled_cost
        step, norm_tier, model_id = task

        reservation_made = False
        reservation_reconciled = False
        estimated_request_cost = 0.0

        if is_live:
            with budget_lock:
                if has_unreconciled_cost:
                    raise RuntimeError(
                        f"Replay run '{run_id}' has unreconciled costs from a previous call (e.g. missing "
                        f"usage metadata or ambiguous failure after dispatch). Further live requests are blocked to prevent untracked spend."
                    )

            if not isinstance(step.request, dict):
                raise ValueError(
                    f"Step {step.id} has invalid or missing request payload. Cannot replay."
                )

            call_kwargs = copy.deepcopy(step.request)
            call_kwargs["model"] = model_id
            call_kwargs.pop("downshift_tag", None)

            max_tokens = call_kwargs.get("max_tokens")
            if max_tokens is None:
                max_tokens = call_kwargs.get("max_completion_tokens")

            if max_tokens is None:
                raise ValueError(
                    f"Missing max_tokens in Step {step.id} request. "
                    "A valid positive output-token limit is required before live dispatch."
                )

            try:
                max_tokens_int = int(max_tokens)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Invalid max_tokens value '{max_tokens}' in Step {step.id} request: must be an integer."
                ) from exc

            if max_tokens_int <= 0:
                raise ValueError(
                    f"Invalid max_tokens value {max_tokens_int} in Step {step.id} request: must be positive."
                )

            est_input_tokens = estimate_input_tokens(call_kwargs)

            with Session(db_mod.engine) as s:
                try:
                    estimated_request_cost = calculate_cost(
                        model_id, est_input_tokens, max_tokens_int, s
                    )
                except (ValueError, KeyError, FileNotFoundError) as exc:
                    raise ValueError(
                        f"Pricing resolution or cost calculation failed for candidate model '{model_id}': {exc}"
                    ) from exc

            with budget_lock:
                if has_unreconciled_cost:
                    raise RuntimeError(
                        f"Replay run '{run_id}' has unreconciled costs from a previous call (e.g. missing "
                        f"usage metadata or ambiguous failure after dispatch). Further live requests are blocked to prevent untracked spend."
                    )
                current_total = cumulative_actual_cost + reserved_cost
                if cost_ceiling_usd is not None:
                    if current_total >= cost_ceiling_usd:
                        raise RuntimeError(
                            f"Cost ceiling of ${cost_ceiling_usd:.6f} reached "
                            f"(current: ${current_total:.6f}). Aborting live replay."
                        )
                    if (current_total + estimated_request_cost) > cost_ceiling_usd:
                        raise RuntimeError(
                            f"Cost ceiling of ${cost_ceiling_usd:.6f} reached: "
                            f"current spent+reserved ${current_total:.6f} plus estimated request cost "
                            f"${estimated_request_cost:.6f} exceeds budget of ${cost_ceiling_usd:.6f}. "
                            "Aborting request before dispatch."
                        )
                reserved_cost += estimated_request_cost
                reservation_made = True
        else:
            # 1. Enforce cost ceiling before making the call in mock mode
            with cost_lock:
                if cost_ceiling_usd is not None and cumulative_cost >= cost_ceiling_usd:
                    raise RuntimeError(
                        f"Cost ceiling of ${cost_ceiling_usd:.6f} reached "
                        f"(current: ${cumulative_cost:.6f}). Aborting replay."
                    )

            # 2. Prepare call kwargs from stored Step.request, replacing ONLY model
            call_kwargs = copy.deepcopy(step.request or {})
            call_kwargs["model"] = model_id

        error_msg: str | None = None
        resp_data: dict[str, Any] | None = None
        tool_calls_data: list[dict[str, Any]] | None = None
        tokens_in = 0
        tokens_out = 0
        latency_ms = 0.0
        cost_usd = 0.0

        t0 = time.perf_counter()
        try:
            raw_response = _execute_with_retry(
                client.chat.completions.create,
                min_wait=retry_min_wait,
                max_wait=retry_max_wait,
                multiplier=retry_multiplier,
                attempts=retry_attempts,
                **call_kwargs,
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0

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

            if usage is None or prompt_tokens is None or completion_tokens is None:
                if usage is None:
                    error_msg = "Response has no usage"
                else:
                    error_msg = "Response has no prompt_tokens or completion_tokens"
                tokens_in = 0
                tokens_out = 0
                cost_usd = 0.0
                if is_live:
                    with budget_lock:
                        has_unreconciled_cost = True
            else:
                tokens_in = int(prompt_tokens)
                tokens_out = int(completion_tokens)
                # Calculate cost via tier's own pricing in database
                with Session(db_mod.engine) as s:
                    cost_usd = calculate_cost(model_id, tokens_in, tokens_out, s)

                if is_live:
                    with budget_lock:
                        cumulative_actual_cost += cost_usd
                        reserved_cost = max(0.0, reserved_cost - estimated_request_cost)
                        reservation_reconciled = True
                else:
                    with cost_lock:
                        cumulative_cost += cost_usd

            choices = (
                raw_response.get("choices", [])
                if isinstance(raw_response, dict)
                else getattr(raw_response, "choices", [])
            )
            if choices:
                msg = (
                    choices[0].get("message")
                    if isinstance(choices[0], dict)
                    else getattr(choices[0], "message", None)
                )
                if msg:
                    tool_calls = (
                        msg.get("tool_calls")
                        if isinstance(msg, dict)
                        else getattr(msg, "tool_calls", None)
                    )
                    if tool_calls:
                        tool_calls_data = _serialize_payload(tool_calls)

            resp_data = _serialize_payload(raw_response)
        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.perf_counter() - t0) * 1000.0
            tokens_in = 0
            tokens_out = 0
            cost_usd = 0.0
            if is_live and reservation_made and not reservation_reconciled:
                # Ambiguous billing: request was already dispatched and encountered an error
                # (e.g. timeout, connection failure, or unexpected provider exception).
                # Provider-side execution/billing is uncertain.
                # Mark cost as unreconciled and DO NOT release the reservation so cumulative
                # budget reflects the conservative upper bound and halts subsequent requests.
                error_msg = (
                    f"{type(exc).__name__}: {exc} [UNRECONCILED_COST: live request dispatched "
                    f"but failed ambiguously; further live requests halted]"
                )
                with budget_lock:
                    has_unreconciled_cost = True
            else:
                error_msg = f"{type(exc).__name__}: {exc}"

        # 3. Persist replay record
        with Session(db_mod.engine) as s:
            old_replays = s.exec(
                select(Replay).where(Replay.step_id == step.id, Replay.tier == norm_tier)
            ).all()
            for old in old_replays:
                # Delete old evaluation first if present, then the replay
                old_evals = s.exec(select(Evaluation).where(Evaluation.replay_id == old.id)).all()
                for old_eval in old_evals:
                    s.delete(old_eval)
                s.delete(old)
            if old_replays:
                s.commit()

            replay_row = Replay(
                step_id=step.id,
                tier=norm_tier,
                model=model_id,
                mode="step_level",
                response=resp_data,
                tool_calls=tool_calls_data,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_ms=latency_ms,
                cost_usd=cost_usd,
                error=error_msg,
            )
            s.add(replay_row)
            s.commit()
            s.refresh(replay_row)

        with replays_lock:
            replays_created.append(replay_row)

        return replay_row

    if is_live:
        for task in tasks:
            with budget_lock:
                if has_unreconciled_cost:
                    raise RuntimeError(
                        f"Replay run '{run_id}' has unreconciled costs from a previous call (e.g. missing "
                        f"usage metadata or ambiguous failure after dispatch). Further live requests are blocked to prevent untracked spend."
                    )
            _execute_single(task)
    else:
        # Execute tasks with ThreadPoolExecutor
        from concurrent.futures import ThreadPoolExecutor

        worker_count = max(1, min(max_workers, len(tasks)))
        executor_error: Exception | None = None

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(_execute_single, t) for t in tasks]
            for f in futures:
                try:
                    f.result()
                except Exception as err:  # noqa: BLE001
                    if executor_error is None:
                        executor_error = err

        if executor_error is not None:
            raise executor_error

    return replays_created

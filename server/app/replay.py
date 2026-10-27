"""Replay engine for executing recorded agent steps across cheaper model tiers."""

import copy
import json
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

import httpx
import openai
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


LIVE_CLIENT_TYPES = (
    openai.OpenAI,
    openai.AsyncOpenAI,
    httpx.Client,
    httpx.AsyncClient,
)


def _contains_live_client(obj: Any, visited: set[int] | None = None) -> bool:
    """Recursively inspect an object to detect if it wraps or contains a live network client."""
    if obj is None:
        return False
    if visited is None:
        visited = set()
    obj_id = id(obj)
    if obj_id in visited:
        return False
    visited.add(obj_id)

    # 1. Direct type check against known network clients
    if isinstance(obj, LIVE_CLIENT_TYPES):
        return True

    # 2. Check for network credentials/endpoints commonly used on OpenAI-compatible clients
    if getattr(obj, "api_key", None) and (
        getattr(obj, "base_url", None) or getattr(obj, "_base_url", None)
    ):
        return True

    # 3. Recursively check object attributes in __slots__ across MRO
    for cls in type(obj).__mro__:
        slots = getattr(cls, "__slots__", None)
        if slots:
            if isinstance(slots, str):
                slots = (slots,)
            for slot in slots:
                if hasattr(obj, slot):
                    val = getattr(obj, slot)
                    if _contains_live_client(val, visited):
                        return True

    # 4. Recursively check object attributes in __dict__
    obj_dict = getattr(obj, "__dict__", None)
    if isinstance(obj_dict, dict):
        for val in obj_dict.values():
            if _contains_live_client(val, visited):
                return True

    return False


def _is_authorized_mock(client: Any) -> bool:
    """Determine whether an object is explicitly an authorized in-memory test mock.

    Follows default-deny security boundary: only recognized mock types and
    mock test doubles are accepted as offline mocks. Setting attributes like
    is_mock=True on an arbitrary client or wrapping a live client is rejected.
    """
    if client is None:
        return True

    # If the object directly is or contains a live client, it can NEVER be an authorized mock
    if _contains_live_client(client):
        return False

    # Standard unittest/pytest mock doubles: verify genuine mock base classes
    import unittest.mock

    if isinstance(client, (unittest.mock.NonCallableMock, unittest.mock.Base)):
        return True

    # In-memory mock classes defined specifically for Downshift
    if isinstance(client, MockReplayClient):
        return True

    try:
        from server.app.judge import MockJudgeClient

        if isinstance(client, MockJudgeClient):
            return True
    except ImportError:
        pass

    return False


def _is_live_client(client: Any) -> bool:
    """Check if client is a live or untrusted client requiring verified live pricing.

    Default-deny principle: Any client that is not explicitly an authorized
    offline in-memory mock is treated as a live client.
    """
    if client is None:
        return False
    return not _is_authorized_mock(client)


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
    execution_mode: str = "offline",
) -> list[Replay]:
    """Replay recorded run steps across candidate tiers without using the recorder.

    Supports explicit execution_mode ('offline' default, 'live' requires verified pricing).
    """
    create_db_and_tables()
    sync_prices_from_yaml()

    if client is not None and _is_live_client(client):
        if execution_mode != "live":
            raise RuntimeError(
                f"REFUSAL: Cannot execute replay against live provider client: execution_mode is '{execution_mode}'. "
                "Explicitly set execution_mode='live' and verify pricing in prices.yaml to authorize live inference."
            )
        is_verified, reason = db_mod.check_live_pricing_verified()
        if not is_verified:
            raise RuntimeError(
                f"REFUSAL: Cannot execute replay against live provider client: {reason}"
            )

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

    if client is None:
        client = MockReplayClient()

    cumulative_cost = 0.0
    cost_lock = threading.Lock()
    replays_created: list[Replay] = []
    replays_lock = threading.Lock()

    def _execute_single(task: tuple[Step, str, str]) -> Replay:
        nonlocal cumulative_cost
        step, norm_tier, model_id = task

        # 1. Enforce cost ceiling before making the call
        with cost_lock:
            if cost_ceiling_usd is not None and cumulative_cost >= cost_ceiling_usd:
                raise RuntimeError(
                    f"Cost ceiling of ${cost_ceiling_usd:.6f} reached "
                    f"(current: ${cumulative_cost:.6f}). Aborting replay."
                )

        # 2. Prepare call kwargs from stored Step.request, replacing ONLY model
        call_kwargs = copy.deepcopy(step.request)
        call_kwargs["model"] = model_id
        # Ensure completion tokens are strictly bounded: clamp or set safe ceiling
        stored_max_tokens = call_kwargs.get("max_tokens")
        if stored_max_tokens is None or int(stored_max_tokens) <= 0:
            call_kwargs["max_tokens"] = 600
        else:
            call_kwargs["max_tokens"] = min(int(stored_max_tokens), 4096)

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

            if usage is None:
                error_msg = "Response has no usage"
                tokens_in = 0
                tokens_out = 0
                cost_usd = 0.0
            elif prompt_tokens is None or completion_tokens is None:
                error_msg = "Response has no prompt_tokens or completion_tokens"
                tokens_in = 0
                tokens_out = 0
                cost_usd = 0.0
            else:
                tokens_in = int(prompt_tokens)
                tokens_out = int(completion_tokens)
                # Calculate cost via tier's own pricing in database
                with Session(db_mod.engine) as s:
                    cost_usd = calculate_cost(model_id, tokens_in, tokens_out, s)

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
            error_msg = f"{type(exc).__name__}: {exc}"
            tokens_in = 0
            tokens_out = 0
            cost_usd = 0.0

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

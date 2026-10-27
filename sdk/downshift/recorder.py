"""OpenAI-compatible client wrapper and call recorder for Downshift."""

import json
import threading
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from sqlmodel import Session

from server.app import db as db_mod
from server.app.db import calculate_cost, create_db_and_tables, resolve_model_price
from server.app.models import Run, Step

MAX_STEPS_PER_RUN = 40
SECRET_EXACT_DENYLIST = {"api_key", "authorization", "headers", "extra_headers"}


class RunContext:
    """Tracking context for an active agent run.

    NOTE ON COST GUARD SEMANTICS:
    The cost ceiling represents a local estimated-cost budget enforced client-side
    using configured pricing tables. It cannot guarantee the provider's final invoice
    or serve as a server-side hard billing cap on the remote API.
    """

    def __init__(self, run: Run, cost_ceiling_usd: float | None = None) -> None:
        self.run = run
        self.cost_ceiling_usd = cost_ceiling_usd
        self._step_counter = 0
        self.has_unreconciled_cost: bool = False
        self._reserved_cost_usd: float = 0.0
        self._lock = threading.Lock()

    def check_and_reserve(self, prospective_cost: float = 0.0) -> int:
        """Enforce limits, reserve prospective cost, and allocate step index before dispatch."""
        with self._lock:
            if self.has_unreconciled_cost:
                raise RuntimeError(
                    f"Run {self.run.id} has unreconciled costs from a previous call with missing or "
                    f"malformed usage metadata. Further calls are blocked to prevent untracked spend."
                )

            if self._step_counter >= MAX_STEPS_PER_RUN:
                raise RuntimeError(
                    f"Maximum step limit ({MAX_STEPS_PER_RUN}) reached. "
                    f"Cannot execute further steps for run {self.run.id}."
                )

            current_total = self.run.total_cost_usd + self._reserved_cost_usd
            if self.cost_ceiling_usd is not None:
                if current_total >= self.cost_ceiling_usd:
                    raise RuntimeError(
                        f"Cost ceiling of ${self.cost_ceiling_usd:.4f} reached "
                        f"(current: ${current_total:.4f}). Aborting call."
                    )
                if (current_total + prospective_cost) > self.cost_ceiling_usd:
                    raise RuntimeError(
                        f"Cost ceiling of ${self.cost_ceiling_usd:.4f} reached: "
                        f"current spent ${current_total:.4f} plus prospective call cost "
                        f"${prospective_cost:.4f} exceeds budget of ${self.cost_ceiling_usd:.4f}. "
                        "Aborting call before dispatch."
                    )

            idx = self._step_counter
            self._step_counter += 1
            self._reserved_cost_usd += prospective_cost
            return idx

    def allocate_step_idx(self, prospective_cost: float = 0.0) -> int:
        """Backwards-compatible wrapper delegating to check_and_reserve."""
        return self.check_and_reserve(prospective_cost=prospective_cost)

    def release_reservation(self, prospective_cost: float) -> None:
        """Release prospective cost reservation after actual cost is reconciled or call fails."""
        with self._lock:
            self._reserved_cost_usd = max(0.0, self._reserved_cost_usd - prospective_cost)


_active_run_ctx: ContextVar[RunContext | None] = ContextVar("active_run_ctx", default=None)


def get_current_run_ctx() -> RunContext | None:
    """Return the active RunContext or None."""
    return _active_run_ctx.get()


@contextmanager
def run(
    task_name: str,
    agent_name: str,
    baseline_model: str = "nemotron_ultra",
    cost_ceiling_usd: float | None = None,
    meta: dict[str, Any] | None = None,
) -> Generator[Run, None, None]:
    """Context manager that creates a Run and updates its totals."""
    create_db_and_tables()
    try:
        db_mod.sync_prices_from_yaml()
    except (FileNotFoundError, ValueError):
        pass

    # Preflight: look up price by model ID / alias, verify not missing or null
    with Session(db_mod.engine) as session:
        real_model_id, price = resolve_model_price(baseline_model, session)
        if price.input_per_mtok is None or price.output_per_mtok is None:
            raise ValueError(
                f"Preflight check failed: Baseline model '{real_model_id}' has unconfigured (null) "
                f"pricing in prices.yaml. Please set official rates before running."
            )

        new_run = Run(
            task_name=task_name,
            agent_name=agent_name,
            baseline_model=real_model_id,
            status="failed",  # Default to failed until completed
            meta=meta,
        )
        session.add(new_run)
        session.commit()
        session.refresh(new_run)

    run_ctx = RunContext(new_run, cost_ceiling_usd=cost_ceiling_usd)
    token = _active_run_ctx.set(run_ctx)

    status = "failed"
    try:
        yield new_run
        status = "completed"
    finally:
        _active_run_ctx.reset(token)
        with Session(db_mod.engine) as session:
            db_run = session.get(Run, new_run.id)
            if db_run:
                db_run.status = status
                session.add(db_run)
                session.commit()


def _serialize_payload(obj: Any) -> Any:
    """Recursively serialize Pydantic models, dicts, or other objects."""
    if type(obj).__name__ in ("MagicMock", "Mock"):
        return str(obj)
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        dump = obj.model_dump()
        if type(dump).__name__ in ("MagicMock", "Mock"):
            return str(obj)
        return _serialize_payload(dump)
    if isinstance(obj, dict):
        return {str(k): _serialize_payload(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize_payload(item) for item in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _sanitize_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Sanitize kwargs by removing downshift_tag and exact denylisted secrets."""
    sanitized: dict[str, Any] = {}
    for k, v in kwargs.items():
        if k == "downshift_tag" or k.lower() in SECRET_EXACT_DENYLIST:
            continue
        sanitized[k] = _serialize_payload(v)
    return sanitized


def record_call(target_func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Intercept, execute, calculate cost, and persist an LLM call step."""
    run_ctx = get_current_run_ctx()
    if not run_ctx:
        raise RuntimeError(
            "No active Downshift run context. Calls to wrapped client must be made inside 'with downshift.run(...)'."
        )

    if kwargs.get("stream", False):
        raise ValueError("Downshift recorder only supports non-streaming calls.")

    max_tokens = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens")
    if max_tokens is None:
        raise ValueError("max_tokens must be explicitly set on every model call.")

    # 1. Conservative preflight estimation of input tokens
    # NOTE: Character-based token estimation is a local heuristic (~4 characters/token
    # plus protocol framing overhead) and cannot guarantee an exact remote billing ceiling.
    messages = kwargs.get("messages", [])
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

    # Include serialized tool definitions passed in kwargs["tools"]
    tools = kwargs.get("tools")
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

    # Protocol framing overhead (~50 tokens) plus ~4 characters per token
    framing_tokens = 50 if (isinstance(messages, list) and messages) or tools else 0
    est_input_tokens = max(1, (approx_chars // 4) + framing_tokens)

    # 2. Preflight pricing resolution and prospective cost estimation
    requested_model = kwargs.get("model") or run_ctx.run.baseline_model
    with Session(db_mod.engine) as session:
        try:
            real_model_id, price = resolve_model_price(requested_model, session)
        except (ValueError, KeyError):
            real_model_id, price = resolve_model_price(run_ctx.run.baseline_model, session)

        if price.input_per_mtok is None or price.output_per_mtok is None:
            raise ValueError(
                f"Pricing for model '{real_model_id}' is unconfigured or contains null values. "
                "Cannot safely verify cost budget."
            )
        prospective_cost = calculate_cost(real_model_id, est_input_tokens, max_tokens, session)

    # 3. Reserve budget and allocate step index before dispatch
    step_idx = run_ctx.check_and_reserve(prospective_cost)

    step_tag = kwargs.pop("downshift_tag", "default")
    request_data = _sanitize_kwargs(kwargs)

    try:
        t0 = time.perf_counter()
        try:
            response = target_func(*args, **kwargs)
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            is_timeout = (
                isinstance(exc, TimeoutError)
                or "timeout" in type(exc).__name__.lower()
                or "timed out" in str(exc).lower()
            )
            if is_timeout:
                run_ctx.has_unreconciled_cost = True
            with Session(db_mod.engine) as session:
                step_record = Step(
                    run_id=run_ctx.run.id,
                    idx=step_idx,
                    tag=step_tag,
                    kind="error",
                    model=real_model_id,
                    request=request_data,
                    response={"error": str(exc), "error_type": type(exc).__name__},
                    tokens_in=0,
                    tokens_out=0,
                    latency_ms=latency_ms,
                    cost_usd=0.0,
                )
                session.add(step_record)
                if is_timeout:
                    db_run = session.get(Run, run_ctx.run.id)
                    if db_run:
                        db_run.status = "failed"
                        meta = dict(db_run.meta or {})
                        meta["has_unreconciled_cost"] = True
                        meta["unreconciled_reason"] = (
                            f"Call timed out ({type(exc).__name__}: {exc}). "
                            "Provider charges are unknown and unreconciled."
                        )
                        meta["estimated_tokens_in"] = est_input_tokens
                        meta["estimated_tokens_out"] = max_tokens
                        meta["estimated_cost_usd"] = prospective_cost
                        db_run.meta = meta
                        session.add(db_run)
                session.commit()
            raise

        latency_ms = (time.perf_counter() - t0) * 1000.0

        usage = getattr(response, "usage", None)
        tokens_in = getattr(usage, "prompt_tokens", None) if usage else None
        tokens_out = getattr(usage, "completion_tokens", None) if usage else None

        # 4. Handle missing or malformed usage metadata safely
        if usage is None or tokens_in is None or tokens_out is None:
            run_ctx.has_unreconciled_cost = True
            with Session(db_mod.engine) as session:
                step_record = Step(
                    run_id=run_ctx.run.id,
                    idx=step_idx,
                    tag=step_tag,
                    kind="error",
                    model=real_model_id,
                    request=request_data,
                    response=_serialize_payload(response),
                    tokens_in=0,
                    tokens_out=0,
                    latency_ms=latency_ms,
                    cost_usd=0.0,
                )
                session.add(step_record)
                db_run = session.get(Run, run_ctx.run.id)
                if db_run:
                    db_run.status = "failed"
                    meta = dict(db_run.meta or {})
                    meta["has_unreconciled_cost"] = True
                    meta["unreconciled_reason"] = (
                        "Provider response missing or malformed usage metadata. "
                        "Exact token counts and charges are unknown."
                    )
                    meta["estimated_tokens_in"] = est_input_tokens
                    meta["estimated_tokens_out"] = max_tokens
                    meta["estimated_cost_usd"] = prospective_cost
                    db_run.meta = meta
                    session.add(db_run)
                session.commit()

            raise ValueError(
                f"Response usage metadata is absent or malformed from model '{real_model_id}'. "
                f"Run '{run_ctx.run.id}' marked as failed with unreconciled cost."
            )

        response_model = getattr(response, "model", None)
        with Session(db_mod.engine) as session:
            final_model_id = real_model_id
            if response_model and response_model != real_model_id:
                try:
                    final_model_id, _ = resolve_model_price(response_model, session)
                except (ValueError, KeyError):
                    final_model_id = real_model_id

            cost_usd = calculate_cost(final_model_id, tokens_in, tokens_out, session)

            choices = getattr(response, "choices", [])
            kind = "text"
            tool_calls_data = None
            if choices:
                message = getattr(choices[0], "message", None)
                if message and getattr(message, "tool_calls", None):
                    kind = "tool_call"
                    tool_calls_data = _serialize_payload(message.tool_calls)

            response_data = _serialize_payload(response)
            step_record = Step(
                run_id=run_ctx.run.id,
                idx=step_idx,
                tag=step_tag,
                kind=kind,
                model=final_model_id,
                request=request_data,
                response=response_data,
                tool_calls=tool_calls_data,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_ms=latency_ms,
                cost_usd=cost_usd,
            )
            session.add(step_record)

            db_run = session.get(Run, run_ctx.run.id)
            if db_run:
                db_run.tokens_in += tokens_in
                db_run.tokens_out += tokens_out
                db_run.total_cost_usd += cost_usd
                session.add(db_run)
                run_ctx.run.total_cost_usd = db_run.total_cost_usd
                run_ctx.run.tokens_in = db_run.tokens_in
                run_ctx.run.tokens_out = db_run.tokens_out

            session.commit()

        return response
    finally:
        run_ctx.release_reservation(prospective_cost)


class _CompletionsProxy:
    """Proxy for client.chat.completions to intercept create calls."""

    def __init__(self, original_completions: Any) -> None:
        self._original = original_completions

    def create(self, *args: Any, **kwargs: Any) -> Any:
        return record_call(self._original.create, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class _ChatProxy:
    """Proxy for client.chat."""

    def __init__(self, original_chat: Any) -> None:
        self._original = original_chat
        self.completions = _CompletionsProxy(original_chat.completions)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class ClientWrapper:
    """Wrapper around any OpenAI-compatible client."""

    def __init__(self, original_client: Any) -> None:
        self._original = original_client
        self.chat = _ChatProxy(original_client.chat)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


def wrap_client(client: Any) -> Any:
    """Wrap an OpenAI-compatible client with one line to record calls."""
    return ClientWrapper(client)

"""Objective evaluation checks for model replay responses."""

import json
from collections.abc import Callable
from typing import Any

from sqlmodel import Session, col, select

from server.app import db as db_mod
from server.app.db import create_db_and_tables
from server.app.models import Evaluation, Replay, Step

TYPE_CHECKERS: dict[str, Callable[[Any], bool]] = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
    "null": lambda v: v is None,
}


def _find_tool_schema(tools: list[dict[str, Any]], tool_name: str) -> dict[str, Any] | None:
    """Find the parameters JSON schema for a given tool name in tools list."""
    for tool in tools:
        fn = tool.get("function", tool)
        if fn.get("name") == tool_name:
            params = fn.get("parameters")
            if isinstance(params, dict):
                return params
    return None


def _validate_json_schema(
    args: dict[str, Any],
    schema: dict[str, Any],
    tool_name: str,
) -> tuple[bool, str | None]:
    """Validate tool arguments dict against a JSON schema parameters definition."""
    # 1. Required keys check
    required_keys = schema.get("required", [])
    if isinstance(required_keys, list):
        for req in required_keys:
            if req not in args:
                return (
                    False,
                    f"Tool '{tool_name}' argument validation failed: missing required parameter '{req}'.",
                )

    # 2. Properties type check and unknown keys check
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        properties = {}

    additional_properties = schema.get("additionalProperties", True)

    for key, val in args.items():
        if key in properties:
            prop_def = properties[key]
            expected_type = prop_def.get("type") if isinstance(prop_def, dict) else None
            if isinstance(expected_type, str):
                checker = TYPE_CHECKERS.get(expected_type)
                if checker and not checker(val):
                    msg = (
                        f"Tool '{tool_name}' argument '{key}' expected type '{expected_type}', "
                        f"got '{type(val).__name__}'."
                    )
                    return False, msg
            elif isinstance(expected_type, list):
                if not any(TYPE_CHECKERS.get(t, lambda _: False)(val) for t in expected_type):
                    msg = (
                        f"Tool '{tool_name}' argument '{key}' expected one of {expected_type}, "
                        f"got '{type(val).__name__}'."
                    )
                    return False, msg
        else:
            if additional_properties is False:
                msg = (
                    f"Tool '{tool_name}' received unknown parameter '{key}' while "
                    "additionalProperties is false."
                )
                return False, msg

    return True, None


def check_replay(step: Step, replay: Replay) -> tuple[bool, list[dict[str, Any]]]:
    """Evaluate a replay against its original step and return (objective_pass, checks)."""
    checks: list[dict[str, Any]] = []

    # 1. Any replay with an error fails.
    if replay.error is not None:
        checks.append(
            {
                "name": "no_replay_error",
                "passed": False,
                "reason": f"Replay failed with error: {replay.error}",
            }
        )
        return False, checks

    checks.append(
        {
            "name": "no_replay_error",
            "passed": True,
            "reason": "Replay completed without recorded errors.",
        }
    )

    if not replay.response or not isinstance(replay.response, dict):
        checks.append(
            {
                "name": "response_present",
                "passed": False,
                "reason": "Replay has no response payload.",
            }
        )
        return False, checks

    choices = replay.response.get("choices", [])
    if not choices or not isinstance(choices, list):
        checks.append(
            {
                "name": "choice_present",
                "passed": False,
                "reason": "Replay response contains no choices.",
            }
        )
        return False, checks

    choice = choices[0] if isinstance(choices[0], dict) else {}
    finish_reason = choice.get("finish_reason")
    message = choice.get("message", {}) if isinstance(choice.get("message"), dict) else {}

    # Check whether the original step was a tool_call step
    is_tool_call_step = (step.kind == "tool_call") or bool(step.tool_calls)

    if is_tool_call_step:
        # 3. Tool call checks

        # A. Finish reason must be 'tool_calls'
        finish_pass = finish_reason == "tool_calls"
        checks.append(
            {
                "name": "finish_reason_tool_calls",
                "passed": finish_pass,
                "reason": (
                    "finish_reason is 'tool_calls'."
                    if finish_pass
                    else f"Expected finish_reason 'tool_calls', got '{finish_reason}'."
                ),
            }
        )

        # B. Same tool names in the same order
        orig_tool_calls = step.tool_calls or []
        if not orig_tool_calls and step.response and isinstance(step.response, dict):
            step_choices = step.response.get("choices", [])
            if step_choices and isinstance(step_choices[0], dict):
                step_msg = step_choices[0].get("message", {})
                orig_tool_calls = step_msg.get("tool_calls") or []

        replay_tool_calls = replay.tool_calls or message.get("tool_calls") or []

        orig_names = [
            tc.get("function", {}).get("name")
            for tc in orig_tool_calls
            if isinstance(tc, dict) and isinstance(tc.get("function"), dict)
        ]
        replay_names = [
            tc.get("function", {}).get("name")
            for tc in replay_tool_calls
            if isinstance(tc, dict) and isinstance(tc.get("function"), dict)
        ]

        names_pass = bool(orig_names) and (orig_names == replay_names)
        checks.append(
            {
                "name": "tool_names_and_order",
                "passed": names_pass,
                "reason": (
                    f"Tool names and order match expected: {orig_names}."
                    if names_pass
                    else f"Tool names or order differ: expected {orig_names}, got {replay_names}."
                ),
            }
        )

        # C. Arguments parse as JSON
        parsed_replay_args: list[dict[str, Any]] = []
        json_pass = True
        json_reason = "Tool call arguments parsed as valid JSON."

        if not replay_tool_calls and orig_names:
            json_pass = False
            json_reason = "No tool calls found on replay response."
        else:
            for tc in replay_tool_calls:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                raw_args = fn.get("arguments", "")
                if isinstance(raw_args, dict):
                    parsed_replay_args.append(raw_args)
                elif isinstance(raw_args, str):
                    try:
                        parsed = json.loads(raw_args)
                        if not isinstance(parsed, dict):
                            json_pass = False
                            json_reason = (
                                f"Tool '{fn.get('name')}' arguments JSON is not an object."
                            )
                            break
                        parsed_replay_args.append(parsed)
                    except Exception as err:  # noqa: BLE001
                        json_pass = False
                        json_reason = f"Tool '{fn.get('name')}' arguments failed JSON parse: {err}."
                        break
                else:
                    json_pass = False
                    json_reason = f"Tool '{fn.get('name')}' arguments has unsupported type: {type(raw_args).__name__}."
                    break

        checks.append(
            {
                "name": "arguments_valid_json",
                "passed": json_pass,
                "reason": json_reason,
            }
        )

        # D. Validate against tool JSON schema
        schema_pass = True
        schema_reason = "Tool arguments comply with tool parameters JSON schema."

        if json_pass and replay_tool_calls:
            tools_def = step.request.get("tools", []) if isinstance(step.request, dict) else []
            for tc, args in zip(replay_tool_calls, parsed_replay_args, strict=False):
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                tool_name = fn.get("name", "")
                schema = _find_tool_schema(tools_def, tool_name)
                if schema is None:
                    schema_pass = False
                    schema_reason = f"Schema definition for tool '{tool_name}' not found in original request tools."
                    break
                valid, err_msg = _validate_json_schema(args, schema, tool_name)
                if not valid:
                    schema_pass = False
                    schema_reason = err_msg or "Schema validation failed."
                    break
        else:
            if not json_pass:
                schema_pass = False
                schema_reason = "Skipped schema validation due to invalid tool arguments JSON."

        checks.append(
            {
                "name": "arguments_schema_validation",
                "passed": schema_pass,
                "reason": schema_reason,
            }
        )

        # E. Arguments equality (information only)
        orig_parsed_args: list[dict[str, Any]] = []
        for tc in orig_tool_calls:
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            raw_args = fn.get("arguments", "")
            if isinstance(raw_args, dict):
                orig_parsed_args.append(raw_args)
            elif isinstance(raw_args, str):
                try:
                    p = json.loads(raw_args)
                    if isinstance(p, dict):
                        orig_parsed_args.append(p)
                except (json.JSONDecodeError, TypeError):
                    pass

        args_equal = bool(parsed_replay_args) and (parsed_replay_args == orig_parsed_args)
        checks.append(
            {
                "name": "arguments_equality_info",
                "passed": True,  # Recorded as information only; does not fail objective_pass
                "reason": (
                    "Replay argument values match original step exactly."
                    if args_equal
                    else "Replay argument values differ legitimately from original step."
                ),
                "arguments_equal": args_equal,
            }
        )

    else:
        # 4. Text step checks
        content = message.get("content")
        reasoning_content = (
            message.get("reasoning_content") or message.get("reasoning") or message.get("thought")
        )

        has_content = bool(content and isinstance(content, str) and content.strip() != "")
        has_reasoning = bool(
            reasoning_content
            and isinstance(reasoning_content, str)
            and reasoning_content.strip() != ""
        )

        if not has_content and has_reasoning:
            content_pass = False
            content_reason = "Answer exists only in reasoning field (content is empty)."
        elif not has_content:
            content_pass = False
            content_reason = "Response content is empty."
        else:
            content_pass = True
            content_reason = "Response content is non-empty."

        checks.append(
            {
                "name": "content_non_empty",
                "passed": content_pass,
                "reason": content_reason,
            }
        )

        finish_pass = finish_reason == "stop"
        checks.append(
            {
                "name": "finish_reason_stop",
                "passed": finish_pass,
                "reason": (
                    "finish_reason is 'stop'."
                    if finish_pass
                    else f"Expected finish_reason 'stop', got '{finish_reason}' (length ending fails)."
                ),
            }
        )

    objective_pass = all(c["passed"] for c in checks)
    return objective_pass, checks


def evaluate_replay(
    step: Step,
    replay: Replay,
    session: Session,
) -> Evaluation:
    """Evaluate a single replay objectively and save/update an Evaluation row."""
    objective_pass, checks = check_replay(step, replay)

    existing = session.exec(select(Evaluation).where(Evaluation.replay_id == replay.id)).first()

    if existing:
        existing.objective_pass = objective_pass
        existing.objective_detail = {"checks": checks}
        session.add(existing)
        session.commit()
        session.refresh(existing)
        return existing

    eval_record = Evaluation(
        replay_id=replay.id,
        objective_pass=objective_pass,
        objective_detail={"checks": checks},
    )
    session.add(eval_record)
    session.commit()
    session.refresh(eval_record)
    return eval_record


def evaluate_run(
    run_id: str,
    session: Session | None = None,
) -> list[Evaluation]:
    """Evaluate all replays for a run and store Evaluation rows without duplicates."""
    create_db_and_tables()
    close_session = False
    if session is None:
        session = Session(db_mod.engine)
        close_session = True

    try:
        steps = session.exec(
            select(Step).where(Step.run_id == run_id).order_by(col(Step.idx))
        ).all()

        evaluations: list[Evaluation] = []
        for step in steps:
            replays = session.exec(select(Replay).where(Replay.step_id == step.id)).all()
            for replay in replays:
                eval_record = evaluate_replay(step, replay, session)
                evaluations.append(eval_record)

        return evaluations
    finally:
        if close_session:
            session.close()

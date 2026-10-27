"""Single-request smoke test script for Nebius Token Factory models via Downshift wrapper.

Executes a single bounded tool-call test on the configured Nemotron Nano model with:
- max_tokens = 15 (strictly bounded completion length)
- trivial get_status tool definition
- tool_choice = 'auto'
- cost ceiling = $0.01 (client-side pre-dispatch estimate, NOT a provider-enforced spending cap)

Prints: finish_reason, tool_calls presence, tool name & arguments, tokens in/out,
cost, and reasoning field names.
Outputs a clear PASS/FAIL line.

Refuses to run if any key, base URL, model ID, or price is missing, placeholder,
or if verified_for_live is false in prices.yaml.
DO NOT run automatically; intended for manual user execution.
"""

import sys
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Session, select

from sdk import downshift
from server.app import db as db_mod
from server.app.cli import inspect_run
from server.app.config import settings
from server.app.models import Run, Step

REASONING_FIELD_NAMES = ("reasoning_content", "reasoning", "thought", "thinking")

TIER_ALIASES = [
    ("ultra", "nemotron_ultra"),
    ("super", "nemotron_super"),
    ("nano", "nemotron_nano"),
]


def validate_environment() -> dict[str, str]:
    """Validate API credentials, base URL, and pricing for Ultra, Super, and Nano tiers."""
    key = settings.nebius_api_key
    if not key or "PLACEHOLDER" in key.upper() or key.startswith("your_"):
        print(
            "REFUSAL: NEBIUS_API_KEY is not configured or contains placeholder text in .env.",
            file=sys.stderr,
        )
        sys.exit(1)

    url = settings.nebius_base_url
    if not url or "PLACEHOLDER" in url.upper():
        print(
            "REFUSAL: NEBIUS_BASE_URL is not configured or contains placeholder text in .env.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Ensure database schema exists before syncing or querying
    db_mod.create_db_and_tables()

    # Sync pricing from prices.yaml into DB
    try:
        db_mod.sync_prices_from_yaml()
    except (FileNotFoundError, ValueError) as e:
        print(f"REFUSAL: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate that pricing has been explicitly verified for live use
    is_verified, reason = db_mod.check_live_pricing_verified()
    if not is_verified:
        print(f"REFUSAL: {reason}", file=sys.stderr)
        sys.exit(1)

    resolved_tiers: dict[str, str] = {}

    with Session(db_mod.engine) as session:
        for tier_label, default_alias in TIER_ALIASES:
            # Check if settings override is provided, else fall back to alias
            model_key = getattr(settings, f"{tier_label}_model_id", None) or default_alias
            try:
                real_model_id, price = db_mod.resolve_model_price(model_key, session)
            except (ValueError, KeyError) as e:
                print(
                    f"REFUSAL: Failed to resolve pricing for {tier_label.capitalize()} tier: {e}",
                    file=sys.stderr,
                )
                sys.exit(1)

            if not real_model_id or "PLACEHOLDER" in real_model_id.upper():
                print(
                    f"REFUSAL: {tier_label.capitalize()} model ID '{real_model_id}' is still a placeholder in prices.yaml.",
                    file=sys.stderr,
                )
                sys.exit(1)

            if price.input_per_mtok is None or price.output_per_mtok is None:
                print(
                    f"REFUSAL: Pricing for {tier_label.capitalize()} model '{real_model_id}' contains null values in prices.yaml.",
                    file=sys.stderr,
                )
                sys.exit(1)

            resolved_tiers[tier_label] = real_model_id

    return resolved_tiers


def _detect_reasoning_fields(message: Any) -> list[str]:
    """Detect non-empty reasoning fields on the response message without reading the text."""
    detected: list[str] = []
    for field in REASONING_FIELD_NAMES:
        val = getattr(message, field, None)
        if val is None and hasattr(message, "model_dump"):
            dump = message.model_dump()
            val = dump.get(field)
        if val:
            detected.append(field)
    return detected


def run_smoke_test() -> None:
    """Execute single tool-call check for Nano tier and print detailed results."""
    resolved_tiers = validate_environment()
    nano_model_id = resolved_tiers["nano"]

    print("=" * 70)
    print("Downshift Single-Request Smoke Test: Nano Tier Only")
    print(f"Base URL   : {settings.nebius_base_url}")
    print(f"Nano Model : {nano_model_id}")
    print(
        "Note: $0.01 ceiling is a client-side estimated cost limit, "
        "not a provider-enforced spending cap."
    )
    print("=" * 70)

    raw_client = OpenAI(
        api_key=settings.nebius_api_key,
        base_url=settings.nebius_base_url,
        timeout=settings.nebius_request_timeout,
        max_retries=settings.nebius_max_retries,
    )
    client = downshift.wrap_client(raw_client)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_status",
                "description": "Get current operational status for a service.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "service": {
                            "type": "string",
                            "description": "The service name, e.g. 'nebius'",
                        }
                    },
                    "required": ["service"],
                },
            },
        }
    ]

    tier_results: dict[str, bool] = {}

    with downshift.run(
        task_name="smoke_test",
        agent_name="smoke_tester",
        baseline_model=nano_model_id,
        cost_ceiling_usd=0.01,
    ) as active_run:
        tier_label = "nano"
        model_id = nano_model_id
        print(f"\n--- Testing Tier: {tier_label.upper()} ({model_id}) ---")

        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[
                    {
                        "role": "user",
                        "content": "Check the status of service 'nebius' using the get_status tool.",
                    }
                ],
                tools=tools,
                tool_choice="auto",
                max_tokens=15,
                downshift_tag=f"{tier_label}_tool_call",
            )
        except (APITimeoutError, APIConnectionError) as err:
            print(
                f"[TIMEOUT_ERROR] Tier {tier_label.capitalize()}: Network connection or timeout error: {err}. "
                "Remote provider charges are UNKNOWN/UNRECONCILED (not confirmed $0.00).",
                file=sys.stderr,
            )
            tier_results[tier_label] = False
        except APIStatusError as err:
            print(
                f"[PROVIDER_ERROR] Tier {tier_label.capitalize()}: HTTP {err.status_code} provider error: {err.message}. "
                "Remote provider charges are UNKNOWN/UNRECONCILED (not confirmed $0.00).",
                file=sys.stderr,
            )
            tier_results[tier_label] = False
        except ValueError as err:
            print(
                f"[USAGE_TRACKING_ERROR] Tier {tier_label.capitalize()}: Usage tracking or pricing error: {err}. "
                "Cost is UNRESOLVED.",
                file=sys.stderr,
            )
            tier_results[tier_label] = False
        except SQLAlchemyError as err:
            print(
                f"[PERSISTENCE_ERROR] Tier {tier_label.capitalize()}: Database storage error: {err}. "
                "Cost is UNRESOLVED.",
                file=sys.stderr,
            )
            tier_results[tier_label] = False
        except Exception as err:  # noqa: BLE001
            print(
                f"[ERROR] Tier {tier_label.capitalize()}: Unexpected execution error: {err}. "
                "Cost is UNRESOLVED.",
                file=sys.stderr,
            )
            tier_results[tier_label] = False
        else:
            choice = resp.choices[0]
            finish_reason = getattr(choice, "finish_reason", "unknown")
            msg = choice.message
            tool_calls = getattr(msg, "tool_calls", None)
            has_tool_calls = bool(tool_calls)

            tool_name = None
            tool_args = None
            if has_tool_calls and tool_calls:
                first_call = tool_calls[0]
                tool_name = first_call.function.name
                tool_args = first_call.function.arguments

            # Retrieve recorded step cost and tokens from DB
            with Session(db_mod.engine) as session:
                step_record = session.exec(
                    select(Step)
                    .where(Step.run_id == active_run.id)
                    .where(Step.tag == f"{tier_label}_tool_call")
                ).first()
                step_cost = step_record.cost_usd if step_record else 0.0
                tokens_in = step_record.tokens_in if step_record else 0
                tokens_out = step_record.tokens_out if step_record else 0

            reasoning_fields = _detect_reasoning_fields(msg)

            print(f"Finish Reason          : {finish_reason}")
            print(f"tool_calls Present     : {has_tool_calls}")
            if has_tool_calls:
                print(f"Tool Name              : {tool_name}")
                print(f"Tool Arguments         : {tool_args}")
            print(f"Tokens In / Out        : {tokens_in} in / {tokens_out} out")
            print(f"Step Cost              : ${step_cost:.6f}")
            print(
                f"Reasoning Fields Found : {reasoning_fields if reasoning_fields else 'None detected'}"
            )

            # Determine pass/fail for this tier
            passed = has_tool_calls and tool_name == "get_status"
            tier_results[tier_label] = passed

            if passed:
                print(f"[PASS] Tier {tier_label.capitalize()}: Tool call generated successfully.")
            else:
                print(
                    f"[FAIL] Tier {tier_label.capitalize()}: Tool call missing or incorrect tool called.",
                    file=sys.stderr,
                )

    with Session(db_mod.engine) as session:
        db_run = session.get(Run, active_run.id)
        has_unreconciled = (
            (db_run.meta or {}).get("has_unreconciled_cost", False) if db_run else False
        )

    if has_unreconciled:
        print(
            "\n[USAGE_TRACKING_ERROR] Run contains unreconciled costs from missing/malformed provider usage. "
            "Remote provider charges cannot be confirmed.",
            file=sys.stderr,
        )
        tier_results["reconciled"] = False

    print("\n" + "=" * 70)
    print("Summary of Smoke Test:")
    for tier_label, ok in tier_results.items():
        status_str = "PASS" if ok else "FAIL"
        print(
            f"  Tier {tier_label.upper():<6}: [{status_str}] ({resolved_tiers.get(tier_label, 'N/A')})"
        )

    print("\nInspecting Run Totals via CLI...")
    inspect_run(active_run.id)

    if not all(tier_results.values()):
        print("\nSmoke test failed.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    run_smoke_test()

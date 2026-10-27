"""Read-only Nebius Token Factory model catalog verification utility.

Calls GET /v1/models using the OpenAI-compatible client metadata API.
Does NOT make any inference calls, embedding requests, or state modifications.
"""

import sys
from typing import Any

import yaml
from openai import APIConnectionError, APIStatusError, APITimeoutError, AuthenticationError, OpenAI

from server.app.config import REPO_ROOT, settings


def get_configured_tiers() -> dict[str, dict[str, Any]]:
    """Load configured model IDs and overrides for Ultra, Super, and Nano tiers."""
    prices_path = settings.prices_path
    if not prices_path.is_absolute():
        prices_path = REPO_ROOT / prices_path

    yaml_data: dict[str, Any] = {}
    if prices_path.exists():
        try:
            with open(prices_path, "r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f) or {}
        except Exception:  # noqa: BLE001
            yaml_data = {}

    models_section = yaml_data.get("models", {})
    return {
        "ultra": {
            "alias": "nemotron_ultra",
            "yaml_id": models_section.get("nemotron_ultra", {}).get("model_id"),
            "override_id": settings.ultra_model_id,
        },
        "super": {
            "alias": "nemotron_super",
            "yaml_id": models_section.get("nemotron_super", {}).get("model_id"),
            "override_id": settings.super_model_id,
        },
        "nano": {
            "alias": "nemotron_nano",
            "yaml_id": models_section.get("nemotron_nano", {}).get("model_id"),
            "override_id": settings.nano_model_id,
        },
    }


def check_nebius_catalog(client: OpenAI | None = None) -> int:
    """Verify visible models on Nebius Token Factory via GET /v1/models.

    Returns 0 on success (all configured models visible), 1 on refusal or failure.
    """
    key = settings.nebius_api_key
    if not key or "PLACEHOLDER" in key.upper() or key.startswith("your_"):
        print(
            "REFUSAL: NEBIUS_API_KEY is not configured or contains placeholder text in .env. "
            "Cannot check catalog without credentials.",
            file=sys.stderr,
        )
        return 1

    url = settings.nebius_base_url
    if not url or "PLACEHOLDER" in url.upper():
        print(
            "REFUSAL: NEBIUS_BASE_URL is not configured or contains placeholder text in .env.",
            file=sys.stderr,
        )
        return 1

    timeout = min(settings.nebius_request_timeout, 15.0)

    print("=" * 70)
    print("Nebius Token Factory: Model Catalog Verification (GET /v1/models)")
    print("=" * 70)
    print(f"Configured Base URL : {url}")
    print("Base URL Source     : settings.nebius_base_url (via .env / NEBIUS_BASE_URL)")
    print(f"Client Timeout      : {timeout:.1f}s (max_retries=0)")
    print("Endpoint Target     : GET /v1/models (Metadata query only; ZERO inference calls)")
    print("=" * 70)

    if client is None:
        client = OpenAI(
            api_key=key,
            base_url=url,
            timeout=timeout,
            max_retries=0,
        )

    try:
        response = client.models.list()
    except AuthenticationError as err:
        status = getattr(err, "status_code", 401)
        print(
            f"[AUTH_ERROR] Authentication failed (HTTP {status}): Access denied by provider. "
            "Verify your NEBIUS_API_KEY credentials.",
            file=sys.stderr,
        )
        return 1
    except (APITimeoutError, APIConnectionError) as err:
        print(
            f"[NETWORK_ERROR] Network connection or timeout error reaching {url}: {err}",
            file=sys.stderr,
        )
        return 1
    except APIStatusError as err:
        print(
            f"[PROVIDER_ERROR] HTTP {err.status_code} provider error: {err.message}",
            file=sys.stderr,
        )
        return 1
    except Exception as err:  # noqa: BLE001
        print(
            f"[ERROR] Catalog request failed unexpectedly: {type(err).__name__}: {err}",
            file=sys.stderr,
        )
        return 1

    raw_data = getattr(response, "data", [])
    model_ids = [m.id for m in raw_data if hasattr(m, "id")]

    print(f"\n[OK] Catalog query succeeded. Total visible models: {len(model_ids)}")

    nemotron_ids = [mid for mid in model_ids if "nemotron" in mid.lower()]
    if nemotron_ids:
        print(f"\nVisible NVIDIA Nemotron Models ({len(nemotron_ids)}):")
        for mid in sorted(nemotron_ids):
            print(f"  - {mid}")
    else:
        print("\nNo model IDs containing 'nemotron' found in visible catalog.")

    print("\nComparing Configured Downshift Tiers against Visible Catalog:")
    print("-" * 70)

    configured_tiers = get_configured_tiers()
    all_matched = True

    for tier_label, info in configured_tiers.items():
        yaml_id = info["yaml_id"]
        override_id = info["override_id"]
        effective_id = override_id or yaml_id or "UNCONFIGURED"

        exact_match = effective_id in model_ids
        case_match = None
        if not exact_match:
            for mid in model_ids:
                if mid.lower() == effective_id.lower():
                    case_match = mid
                    break

        if exact_match:
            status_str = "EXACT_MATCH"
        elif case_match:
            status_str = f"CASE_MISMATCH (Catalog has: '{case_match}')"
            all_matched = False
        else:
            status_str = "NOT_FOUND in catalog"
            all_matched = False

        print(f"  Tier: {tier_label.upper():<6}")
        print(f"    Effective ID  : {effective_id}")
        if override_id:
            print(f"    Override ID   : {override_id} (active)")
            print(f"    YAML ID       : {yaml_id}")
        else:
            print(f"    Source        : prices.yaml ({yaml_id})")
        print(f"    Catalog Status: [{status_str}]")

    print("\n" + "=" * 70)
    print("SAFETY NOTICE & LIMITATIONS:")
    print("  1. A catalog match confirms ONLY that an ID is listed/visible to this account.")
    print("  2. It does NOT prove inference permission, model availability under load,")
    print("     official pricing accuracy, or provider billing rates.")
    print("  3. Zero tokens were generated, and zero inference requests were dispatched.")
    print("=" * 70)

    return 0 if all_matched else 1


if __name__ == "__main__":
    sys.exit(check_nebius_catalog())

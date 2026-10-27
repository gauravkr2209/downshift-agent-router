"""Tests for smoke test validation against fresh/empty databases."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from openai import APIStatusError, APITimeoutError

from scripts.smoke_test import _detect_reasoning_fields, run_smoke_test, validate_environment
from server.app.config import settings
from server.app.db import check_live_pricing_verified


def test_smoke_validation_refuses_cleanly_on_fresh_empty_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test that validate_environment creates tables and refuses with a clean message, never 'no such table'."""
    monkeypatch.setattr(settings, "nebius_api_key", "test-key-12345")
    monkeypatch.setattr(settings, "nebius_base_url", "https://api.test.ai/v1")
    monkeypatch.setattr(settings, "ultra_model_id", None)
    monkeypatch.setattr(settings, "super_model_id", None)
    monkeypatch.setattr(settings, "nano_model_id", None)

    # Point to a temporary prices.yaml with placeholder model
    dummy_prices = tmp_path / "prices.yaml"
    dummy_prices.write_text(
        """
models:
  nemotron_ultra:
    model_id: "PLACEHOLDER_ULTRA_ID"
    input_per_mtok: null
    output_per_mtok: null
  nemotron_super:
    model_id: "PLACEHOLDER_SUPER_ID"
    input_per_mtok: null
    output_per_mtok: null
  nemotron_nano:
    model_id: "PLACEHOLDER_NANO_ID"
    input_per_mtok: null
    output_per_mtok: null
"""
    )
    monkeypatch.setattr(settings, "prices_path", dummy_prices)

    with pytest.raises(SystemExit) as exc_info:
        validate_environment()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    # Verify no sqlite 'no such table' traceback occurred
    assert "no such table" not in captured.err.lower()
    assert "Traceback" not in captured.err
    # Verify clear REFUSAL message was printed
    assert "REFUSAL:" in captured.err


def test_smoke_validation_succeeds_when_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that validate_environment succeeds when valid credentials and pricing are provided for all 3 tiers."""
    monkeypatch.setattr(settings, "nebius_api_key", "valid-nebius-key")
    monkeypatch.setattr(settings, "nebius_base_url", "https://api.test.ai/v1")
    monkeypatch.setattr(settings, "ultra_model_id", "nvidia/nemotron-ultra-real")
    monkeypatch.setattr(settings, "super_model_id", "nvidia/nemotron-super-real")
    monkeypatch.setattr(settings, "nano_model_id", "nvidia/nemotron-nano-real")

    dummy_prices = tmp_path / "prices.yaml"
    dummy_prices.write_text(
        """
verified_for_live: true

models:
  nemotron_ultra:
    model_id: "nvidia/nemotron-ultra-real"
    input_per_mtok: 1.5
    output_per_mtok: 3.0
  nemotron_super:
    model_id: "nvidia/nemotron-super-real"
    input_per_mtok: 0.8
    output_per_mtok: 1.6
  nemotron_nano:
    model_id: "nvidia/nemotron-nano-real"
    input_per_mtok: 0.2
    output_per_mtok: 0.4
"""
    )
    monkeypatch.setattr(settings, "prices_path", dummy_prices)

    tiers = validate_environment()
    assert tiers["ultra"] == "nvidia/nemotron-ultra-real"
    assert tiers["super"] == "nvidia/nemotron-super-real"
    assert tiers["nano"] == "nvidia/nemotron-nano-real"


def test_detect_reasoning_fields() -> None:
    """Test that reasoning field names are detected without exposing text."""

    class MockMsg:
        def __init__(self) -> None:
            self.content = "hello"
            self.reasoning_content = "internal thought process"
            self.thought = ""

    msg = MockMsg()
    fields = _detect_reasoning_fields(msg)
    assert fields == ["reasoning_content"]


def test_smoke_client_limits_configured_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that OpenAI client in smoke test is constructed with explicit timeout and max_retries=0."""
    from openai import OpenAI

    monkeypatch.setattr(settings, "nebius_api_key", "mock-key")
    monkeypatch.setattr(settings, "nebius_base_url", "https://api.mock.nebius/v1")
    monkeypatch.setattr(settings, "nebius_request_timeout", 25.0)
    monkeypatch.setattr(settings, "nebius_max_retries", 0)

    client = OpenAI(
        api_key=settings.nebius_api_key,
        base_url=settings.nebius_base_url,
        timeout=settings.nebius_request_timeout,
        max_retries=settings.nebius_max_retries,
    )
    assert client.timeout == 25.0
    assert client.max_retries == 0


def test_smoke_validation_refuses_when_pricing_unverified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify that validate_environment refuses when verified_for_live is false."""
    monkeypatch.setattr(settings, "nebius_api_key", "test-key-12345")
    monkeypatch.setattr(settings, "nebius_base_url", "https://api.test.ai/v1")
    monkeypatch.setattr(settings, "ultra_model_id", "nvidia/nemotron-ultra-real")
    monkeypatch.setattr(settings, "super_model_id", "nvidia/nemotron-super-real")
    monkeypatch.setattr(settings, "nano_model_id", "nvidia/nemotron-nano-real")

    dummy_prices = tmp_path / "prices.yaml"
    dummy_prices.write_text(
        """
verified_for_live: false

models:
  nemotron_ultra:
    model_id: "nvidia/nemotron-ultra-real"
    input_per_mtok: 1.0
    output_per_mtok: 3.0
  nemotron_super:
    model_id: "nvidia/nemotron-super-real"
    input_per_mtok: 0.3
    output_per_mtok: 0.9
  nemotron_nano:
    model_id: "nvidia/nemotron-nano-real"
    input_per_mtok: 0.06
    output_per_mtok: 0.24
"""
    )
    monkeypatch.setattr(settings, "prices_path", dummy_prices)

    with pytest.raises(SystemExit) as exc_info:
        validate_environment()
    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "REFUSAL: prices.yaml has verified_for_live=false" in err


@pytest.mark.parametrize(
    "yaml_content,expected_verified",
    [
        ("verified_for_live: true", True),
        ("verified_for_live: false", False),
        ('verified_for_live: "true"', False),
        ('verified_for_live: "false"', False),
        ("verified_for_live: 1", False),
        ("verified_for_live: 0", False),
        ("verified_for_live: null", False),
        ("unrelated_key: 123", False),
    ],
)
def test_strict_pricing_verification_rejects_non_booleans(
    tmp_path: Path, yaml_content: str, expected_verified: bool
) -> None:
    """Test that check_live_pricing_verified only returns True for strict boolean True."""
    f = tmp_path / "prices_test.yaml"
    f.write_text(yaml_content)
    is_verified, reason = check_live_pricing_verified(f)
    assert is_verified is expected_verified
    if not expected_verified:
        assert "verified_for_live" in reason


def _setup_mock_smoke_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Helper to configure environment and dummy prices for smoke test runs."""
    monkeypatch.setattr(settings, "nebius_api_key", "test-key-live")
    monkeypatch.setattr(settings, "nebius_base_url", "https://api.mock.nebius/v1")
    monkeypatch.setattr(settings, "ultra_model_id", "nvidia/nemotron-ultra-real")
    monkeypatch.setattr(settings, "super_model_id", "nvidia/nemotron-super-real")
    monkeypatch.setattr(settings, "nano_model_id", "nvidia/nemotron-nano-real")

    dummy_prices = tmp_path / "prices.yaml"
    dummy_prices.write_text(
        """
verified_for_live: true

models:
  nemotron_ultra:
    model_id: "nvidia/nemotron-ultra-real"
    input_per_mtok: 1.0
    output_per_mtok: 3.0
  nemotron_super:
    model_id: "nvidia/nemotron-super-real"
    input_per_mtok: 0.3
    output_per_mtok: 0.9
  nemotron_nano:
    model_id: "nvidia/nemotron-nano-real"
    input_per_mtok: 0.06
    output_per_mtok: 0.24
"""
    )
    monkeypatch.setattr(settings, "prices_path", dummy_prices)


def test_smoke_run_provider_error_handling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify that provider errors (APIStatusError) emit [PROVIDER_ERROR] and exit code 1 without network access."""
    _setup_mock_smoke_env(tmp_path, monkeypatch)

    mock_resp = MagicMock()
    mock_resp.status_code = 503
    err = APIStatusError(
        message="Service Unavailable",
        response=mock_resp,
        body={"error": "temporarily unavailable"},
    )

    mock_raw_client = MagicMock()
    mock_raw_client.chat.completions.create.side_effect = err
    monkeypatch.setattr("scripts.smoke_test.OpenAI", lambda **_kwargs: mock_raw_client)

    with pytest.raises(SystemExit) as exc_info:
        run_smoke_test()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "[PROVIDER_ERROR]" in captured.err
    assert "HTTP 503 provider error" in captured.err
    assert "UNKNOWN/UNRECONCILED (not confirmed $0.00)" in captured.err
    assert mock_raw_client.chat.completions.create.call_count == 1


def test_smoke_run_timeout_error_handling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify that timeout errors (APITimeoutError) emit [TIMEOUT_ERROR] and exit code 1 without network access."""
    _setup_mock_smoke_env(tmp_path, monkeypatch)

    err = APITimeoutError(request=MagicMock())

    mock_raw_client = MagicMock()
    mock_raw_client.chat.completions.create.side_effect = err
    monkeypatch.setattr("scripts.smoke_test.OpenAI", lambda **_kwargs: mock_raw_client)

    with pytest.raises(SystemExit) as exc_info:
        run_smoke_test()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "[TIMEOUT_ERROR]" in captured.err
    assert "Network connection or timeout error" in captured.err
    assert "UNKNOWN/UNRECONCILED (not confirmed $0.00)" in captured.err
    assert mock_raw_client.chat.completions.create.call_count == 1


def test_smoke_test_targets_only_nano_tier_with_max_tokens_15(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify smoke test targets only the Nano tier with max_tokens=15 and exactly 1 dispatch."""
    _setup_mock_smoke_env(tmp_path, monkeypatch)

    mock_tool_call = MagicMock()
    mock_tool_call.function.name = "get_status"
    mock_tool_call.function.arguments = '{"service": "nebius"}'

    mock_msg = MagicMock()
    mock_msg.role = "assistant"
    mock_msg.content = None
    mock_msg.tool_calls = [mock_tool_call]
    mock_msg.reasoning_content = None

    mock_choice = MagicMock()
    mock_choice.finish_reason = "tool_calls"
    mock_choice.message = mock_msg

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 20
    mock_usage.completion_tokens = 10
    mock_usage.total_tokens = 30

    mock_resp = MagicMock()
    mock_resp.id = "chatcmpl-nano-test"
    mock_resp.model = "nvidia/nemotron-nano-real"
    mock_resp.choices = [mock_choice]
    mock_resp.usage = mock_usage

    mock_raw_client = MagicMock()
    mock_raw_client.chat.completions.create.return_value = mock_resp
    monkeypatch.setattr("scripts.smoke_test.OpenAI", lambda **_kwargs: mock_raw_client)

    run_smoke_test()

    assert mock_raw_client.chat.completions.create.call_count == 1
    call_args = mock_raw_client.chat.completions.create.call_args
    kwargs = call_args.kwargs
    assert kwargs["model"] == "nvidia/nemotron-nano-real"
    assert kwargs["max_tokens"] == 15
    assert kwargs["tools"][0]["function"]["name"] == "get_status"
    # Ensure downshift_tag is stripped before reaching the underlying provider client
    assert "downshift_tag" not in kwargs
    # Ensure Ultra and Super models are never requested
    assert "ultra" not in kwargs["model"].lower()
    assert "super" not in kwargs["model"].lower()

    captured = capsys.readouterr()
    assert "Downshift Single-Request Smoke Test: Nano Tier Only" in captured.out
    assert "client-side estimated cost limit, not a provider-enforced spending cap" in captured.out
    assert "[PASS] Tier Nano: Tool call generated successfully." in captured.out


def test_smoke_test_refuses_when_pricing_gate_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify smoke test refuses execution without dispatching if verified_for_live is false."""
    _setup_mock_smoke_env(tmp_path, monkeypatch)

    dummy_prices = tmp_path / "prices.yaml"
    dummy_prices.write_text(
        """
verified_for_live: false

models:
  nemotron_ultra:
    model_id: "nvidia/nemotron-ultra-real"
    input_per_mtok: 1.0
    output_per_mtok: 3.0
  nemotron_super:
    model_id: "nvidia/nemotron-super-real"
    input_per_mtok: 0.3
    output_per_mtok: 0.9
  nemotron_nano:
    model_id: "nvidia/nemotron-nano-real"
    input_per_mtok: 0.06
    output_per_mtok: 0.24
"""
    )
    monkeypatch.setattr(settings, "prices_path", dummy_prices)

    mock_raw_client = MagicMock()
    monkeypatch.setattr("scripts.smoke_test.OpenAI", lambda **_kwargs: mock_raw_client)

    with pytest.raises(SystemExit) as exc_info:
        run_smoke_test()

    assert exc_info.value.code == 1
    assert mock_raw_client.chat.completions.create.call_count == 0
    err = capsys.readouterr().err
    assert "REFUSAL: prices.yaml has verified_for_live=false" in err


def test_smoke_test_missing_usage_marks_unreconciled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify that missing usage metadata in response marks run as unreconciled and exits with code 1."""
    _setup_mock_smoke_env(tmp_path, monkeypatch)

    mock_resp = MagicMock()
    mock_resp.id = "chatcmpl-nano-test"
    mock_resp.model = "nvidia/nemotron-nano-real"
    mock_resp.choices = [MagicMock()]
    mock_resp.usage = None  # Missing usage metadata

    mock_raw_client = MagicMock()
    mock_raw_client.chat.completions.create.return_value = mock_resp
    monkeypatch.setattr("scripts.smoke_test.OpenAI", lambda **_kwargs: mock_raw_client)

    with pytest.raises(SystemExit) as exc_info:
        run_smoke_test()

    assert exc_info.value.code == 1
    assert mock_raw_client.chat.completions.create.call_count == 1
    captured = capsys.readouterr()
    assert "[USAGE_TRACKING_ERROR]" in captured.err
    assert "unreconciled" in captured.err.lower()


def test_run_baseline_refuses_when_live_pricing_unverified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify that run_baseline with use_real_api=True refuses execution when verified_for_live is false."""
    from examples.research_agent.run_baseline import run_baseline

    monkeypatch.setattr(settings, "nebius_api_key", "test-key-live")
    monkeypatch.setattr(settings, "nebius_base_url", "https://api.mock.nebius/v1")
    monkeypatch.setattr(settings, "ultra_model_id", "nvidia/nemotron-ultra-real")

    dummy_prices = tmp_path / "prices.yaml"
    dummy_prices.write_text(
        """
verified_for_live: false

models:
  nemotron_ultra:
    model_id: "nvidia/nemotron-ultra-real"
    input_per_mtok: 1.0
    output_per_mtok: 3.0
"""
    )
    monkeypatch.setattr(settings, "prices_path", dummy_prices)

    with pytest.raises(SystemExit) as exc_info:
        run_baseline(runs=1, use_real_api=True)

    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "REFUSAL: prices.yaml has verified_for_live=false" in err


def test_compare_runs_refuses_when_live_pricing_unverified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that compare_research_routing with use_real_api=True refuses execution when verified_for_live is false."""
    from examples.research_agent.compare_runs import compare_research_routing

    monkeypatch.setattr(settings, "nebius_api_key", "test-key-live")
    monkeypatch.setattr(settings, "nebius_base_url", "https://api.mock.nebius/v1")
    monkeypatch.setattr(settings, "ultra_model_id", "nvidia/nemotron-ultra-real")

    dummy_prices = tmp_path / "prices.yaml"
    dummy_prices.write_text(
        """
verified_for_live: false

models:
  nemotron_ultra:
    model_id: "nvidia/nemotron-ultra-real"
    input_per_mtok: 1.0
    output_per_mtok: 3.0
"""
    )
    monkeypatch.setattr(settings, "prices_path", dummy_prices)

    with pytest.raises(ValueError, match="REFUSAL: prices.yaml has verified_for_live=false"):
        compare_research_routing(use_real_api=True)


def test_live_clients_use_configured_zero_retries_and_timeouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that run_baseline and compare_runs instantiate live clients with settings.nebius_max_retries and timeout."""
    from unittest.mock import MagicMock, patch

    from examples.research_agent.compare_runs import compare_research_routing
    from examples.research_agent.run_baseline import run_baseline

    monkeypatch.setattr(settings, "nebius_api_key", "test-key-live")
    monkeypatch.setattr(settings, "nebius_base_url", "https://api.mock.nebius/v1")
    monkeypatch.setattr(settings, "ultra_model_id", "nvidia/Nemotron-3-Ultra-550b-a55b")
    monkeypatch.setattr(settings, "nebius_max_retries", 0)
    monkeypatch.setattr(settings, "nebius_request_timeout", 30.0)

    # Allow mock pricing verification for this test only
    from sqlmodel import Session

    import server.app.db as app_db
    from server.app.models import Run

    monkeypatch.setattr(app_db, "check_live_pricing_verified", lambda: (True, "mock verified"))

    with Session(app_db.engine) as session:
        r = Run(
            task_name="test_live_client",
            agent_name="research_agent",
            baseline_model="nvidia/Nemotron-3-Ultra-550b-a55b",
            status="completed",
        )
        session.add(r)
        session.commit()
        session.refresh(r)
        test_run_id = r.id

    with (
        patch("examples.research_agent.run_baseline.OpenAI") as mock_openai_baseline,
        patch("examples.research_agent.run_baseline.ResearchAgent") as mock_agent_baseline,
    ):
        mock_inst = MagicMock()
        mock_inst.run.return_value = {
            "run_id": test_run_id,
            "checklist_score": 1.0,
            "cost_usd": 0.001,
            "tokens_in": 10,
            "tokens_out": 10,
            "task_success": True,
            "final_answer": "Complete answer",
            "search_engine_name": "Tavily",
            "routing_map": {},
            "evaluation": {
                "checklist_score": 1.0,
                "total_items": 4,
                "passed_items": 4,
                "matched_keywords": {},
                "missing_keywords": {},
            },
        }
        mock_agent_baseline.return_value = mock_inst
        run_baseline(runs=1, use_real_api=True)
        mock_openai_baseline.assert_called_once()
        _, kwargs = mock_openai_baseline.call_args
        assert kwargs.get("max_retries") == 0
        assert kwargs.get("timeout") == 30.0

    with (
        patch("examples.research_agent.compare_runs.OpenAI") as mock_openai_compare,
        patch("examples.research_agent.compare_runs.ResearchAgent") as mock_agent_compare,
    ):
        mock_inst = MagicMock()
        mock_inst.run.return_value = {
            "run_id": test_run_id,
            "checklist_score": 1.0,
            "cost_usd": 0.001,
            "tokens_in": 10,
            "tokens_out": 10,
            "task_success": True,
            "final_answer": "Complete answer",
            "search_engine_name": "Tavily",
            "routing_map": {},
            "evaluation": {
                "checklist_score": 1.0,
                "total_items": 4,
                "passed_items": 4,
                "matched_keywords": {},
                "missing_keywords": {},
            },
        }
        mock_agent_compare.return_value = mock_inst
        compare_research_routing(use_real_api=True, print_output=False)
        mock_openai_compare.assert_called_once()
        _, kwargs = mock_openai_compare.call_args
        assert kwargs.get("max_retries") == 0
        assert kwargs.get("timeout") == 30.0

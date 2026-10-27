"""Tests for read-only Nebius model catalog verification utility."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from openai import APIStatusError, APITimeoutError, AuthenticationError

from scripts.check_nebius_models import check_nebius_catalog
from server.app.config import settings


def test_missing_api_key_refuses_before_network(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify that catalog check refuses immediately when NEBIUS_API_KEY is not set."""
    monkeypatch.setattr(settings, "nebius_api_key", None)
    monkeypatch.setattr(settings, "nebius_base_url", "https://api.test.ai/v1")

    exit_code = check_nebius_catalog()
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "REFUSAL: NEBIUS_API_KEY is not configured" in captured.err


def test_placeholder_api_key_refuses_before_network(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify that placeholder API keys refuse without network access."""
    monkeypatch.setattr(settings, "nebius_api_key", "your_api_key_placeholder")
    monkeypatch.setattr(settings, "nebius_base_url", "https://api.test.ai/v1")

    exit_code = check_nebius_catalog()
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "REFUSAL: NEBIUS_API_KEY is not configured" in captured.err


def test_missing_base_url_refuses_before_network(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify that missing base URL refuses without network access."""
    monkeypatch.setattr(settings, "nebius_api_key", "valid-secret-key-12345")
    monkeypatch.setattr(settings, "nebius_base_url", None)

    exit_code = check_nebius_catalog()
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "REFUSAL: NEBIUS_BASE_URL is not configured" in captured.err


def test_catalog_query_calls_only_models_list_and_handles_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify catalog query calls models.list and compares tiers with exact and case-mismatch reporting."""
    monkeypatch.setattr(settings, "nebius_api_key", "secret-test-key-xyz")
    monkeypatch.setattr(settings, "nebius_base_url", "https://api.mock.nebius/v1")

    dummy_prices = tmp_path / "prices.yaml"
    dummy_prices.write_text(
        """
models:
  nemotron_ultra:
    model_id: "nvidia/Nemotron-3-Ultra-550b-a55b"
    input_per_mtok: 1.0
    output_per_mtok: 3.0
  nemotron_super:
    model_id: "nvidia/nemotron-3-super-120b-a12b"
    input_per_mtok: 0.3
    output_per_mtok: 0.9
  nemotron_nano:
    model_id: "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
    input_per_mtok: 0.06
    output_per_mtok: 0.24
"""
    )
    monkeypatch.setattr(settings, "prices_path", dummy_prices)

    mock_client = MagicMock()
    # Mock models returned by GET /v1/models:
    # 1. Exact match for super: nvidia/nemotron-3-super-120b-a12b
    # 2. Case mismatch for ultra: catalog has lowercase nvidia/nemotron-3-ultra-550b-a55b
    # 3. Nano is absent
    mock_client.models.list.return_value.data = [
        MagicMock(id="nvidia/nemotron-3-super-120b-a12b"),
        MagicMock(id="nvidia/nemotron-3-ultra-550b-a55b"),
        MagicMock(id="meta/llama-3.1-8b-instruct"),
    ]

    exit_code = check_nebius_catalog(client=mock_client)

    # models.list called exactly once; chat completions never touched
    mock_client.models.list.assert_called_once()
    assert not hasattr(mock_client, "chat") or mock_client.chat.completions.create.call_count == 0

    captured = capsys.readouterr()
    assert "Total visible models: 3" in captured.out
    assert "[EXACT_MATCH]" in captured.out
    assert "CASE_MISMATCH (Catalog has: 'nvidia/nemotron-3-ultra-550b-a55b')" in captured.out
    assert "[NOT_FOUND in catalog]" in captured.out
    assert "Zero tokens were generated, and zero inference requests were dispatched" in captured.out

    # Secret key must NEVER appear in output
    assert "secret-test-key-xyz" not in captured.out
    assert "secret-test-key-xyz" not in captured.err

    # exit code is 1 because not all models were exact matches
    assert exit_code == 1


def test_catalog_authentication_error_handled_safely(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify that authentication errors are reported safely without exposing credentials."""
    secret_key = "super-secret-bearer-token"
    monkeypatch.setattr(settings, "nebius_api_key", secret_key)
    monkeypatch.setattr(settings, "nebius_base_url", "https://api.mock.nebius/v1")

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_client.models.list.side_effect = AuthenticationError(
        message="Invalid credentials supplied in request headers",
        response=mock_resp,
        body=None,
    )

    exit_code = check_nebius_catalog(client=mock_client)
    assert exit_code == 1

    captured = capsys.readouterr()
    assert "[AUTH_ERROR]" in captured.err
    assert "HTTP 401" in captured.err
    # Verify no secret leakage
    assert secret_key not in captured.out
    assert secret_key not in captured.err


def test_catalog_timeout_error_handled_safely(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify that timeout errors on catalog endpoint exit cleanly with diagnostic message."""
    monkeypatch.setattr(settings, "nebius_api_key", "secret-test-key")
    monkeypatch.setattr(settings, "nebius_base_url", "https://api.mock.nebius/v1")

    mock_client = MagicMock()
    mock_client.models.list.side_effect = APITimeoutError(request=MagicMock())

    exit_code = check_nebius_catalog(client=mock_client)
    assert exit_code == 1

    captured = capsys.readouterr()
    assert "[NETWORK_ERROR]" in captured.err
    assert "Network connection or timeout error" in captured.err


def test_catalog_provider_error_handled_safely(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify that provider 500 status errors exit cleanly with diagnostic message."""
    monkeypatch.setattr(settings, "nebius_api_key", "secret-test-key")
    monkeypatch.setattr(settings, "nebius_base_url", "https://api.mock.nebius/v1")

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 503
    mock_client.models.list.side_effect = APIStatusError(
        message="Service Unavailable",
        response=mock_resp,
        body=None,
    )

    exit_code = check_nebius_catalog(client=mock_client)
    assert exit_code == 1

    captured = capsys.readouterr()
    assert "[PROVIDER_ERROR]" in captured.err
    assert "HTTP 503 provider error" in captured.err

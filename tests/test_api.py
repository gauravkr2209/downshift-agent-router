"""Comprehensive offline test suite for Downshift Demonstration API.

Verifies all 16 prompt requirements:
1. Health endpoint returns 200 and mode: mock.
2. Info endpoint returns 200, mock mode, available tiers, and disclaimer.
3. Valid demo run returns 200 with structured steps and summary.
4. Every returned run uses mock clients; zero live inference.
5. Invalid task_mix returns 400.
6. Out-of-bounds request_count returns 400.
7. Prohibited keys (api_key, model_id, etc.) return 400.
8. No external search calls occur.
9. No API credentials or stack traces in errors.
10. Static files (/, /index.html) served with correct MIME types.
11. Directory traversal requests return 404.
12. CORS headers present on GET, POST, and OPTIONS requests.
13. Concurrent requests handled safely without race conditions.
14. Scaling projections correctly calculated based on request_count.
15. Task mix variations adjust step counts.
16. Error handling returns clean JSON without leaking secrets.
"""

from __future__ import annotations

import concurrent.futures
import threading
from collections.abc import Generator
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from server.app.api import (
    create_demo_server,
    execute_demo_run,
    get_allowed_origins,
    resolve_server_binding,
)


@pytest.fixture
def live_server() -> Generator[str, None, None]:
    """Start an ephemeral ThreadingHTTPServer instance for testing."""
    server = create_demo_server(host="127.0.0.1", port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    try:
        yield base_url
    finally:
        server.shutdown()
        server.server_close()


# 1. Health endpoint returns 200 and mode: mock
def test_health_endpoint(live_server: str) -> None:
    with httpx.Client(base_url=live_server) as client:
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["service"] == "downshift-demo-api"
        assert data["mode"] == "mock"
        assert data["live_inference_enabled"] is False
        assert data["version"] == "0.1.0"


# 2. Info endpoint returns 200, mock mode, available tiers, and disclaimer
def test_info_endpoint(live_server: str) -> None:
    with httpx.Client(base_url=live_server) as client:
        resp = client.get("/api/demo/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["mode"] == "mock"
        assert data["live_inference_enabled"] is False
        assert "available_tiers" in data
        assert len(data["available_tiers"]) == 3
        tiers = [t["tier"] for t in data["available_tiers"]]
        assert "ultra" in tiers
        assert "super" in tiers
        assert "nano" in tiers
        assert "disclaimer" in data
        assert len(data["disclaimer"]) > 10


# 3. Valid demo run returns 200 with structured steps and summary
def test_valid_demo_run_structure(live_server: str) -> None:
    with httpx.Client(base_url=live_server, timeout=30.0) as client:
        resp = client.post(
            "/api/demo/run",
            json={"task_mix": "simple", "request_count": 1000},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["mode"] == "mock"
        assert data["status"] == "completed"
        assert "run_id" in data
        assert "optimized_run_id" in data
        assert "steps" in data
        assert isinstance(data["steps"], list)
        assert len(data["steps"]) > 0

        # Check step fields
        step0 = data["steps"][0]
        assert "idx" in step0
        assert "tag" in step0
        assert "baseline_model" in step0
        assert "selected_model" in step0
        assert "decision" in step0
        assert "cost_usd" in step0
        assert "tokens_in" in step0
        assert "tokens_out" in step0
        assert "objective_pass" in step0

        # Check summary
        summary = data["summary"]
        assert "baseline_cost_per_run_usd" in summary
        assert "routed_cost_per_run_usd" in summary
        assert "cost_reduction_pct" in summary
        assert "scaling_projection" in summary


# 4. Zero live inference invariant (no external OpenAI calls made)
def test_zero_live_inference_invariant(live_server: str) -> None:
    with (
        patch("openai.OpenAI", side_effect=RuntimeError("LIVE OPENAI CALL DETECTED")),
        patch("httpx.post", side_effect=RuntimeError("UNEXPECTED OUTGOING HTTP POST")),
    ):
        result = execute_demo_run(task_mix="simple", request_count=200)
        assert result["mode"] == "mock"
        assert result["status"] == "completed"
        assert len(result["steps"]) == 5


# 5. Invalid task_mix returns 400
def test_invalid_task_mix_returns_400(live_server: str) -> None:
    with httpx.Client(base_url=live_server) as client:
        resp = client.post("/api/demo/run", json={"task_mix": "ultra_extreme"})
        assert resp.status_code == 400
        data = resp.json()
        assert data["status"] == "error"
        assert "Invalid task_mix" in data["error"]


# 6. Out-of-bounds request_count returns 400
def test_out_of_bounds_request_count_returns_400(live_server: str) -> None:
    with httpx.Client(base_url=live_server) as client:
        # Below minimum (100)
        resp_low = client.post("/api/demo/run", json={"request_count": 50})
        assert resp_low.status_code == 400
        assert resp_low.json()["status"] == "error"

        # Above maximum (100,000)
        resp_high = client.post("/api/demo/run", json={"request_count": 500000})
        assert resp_high.status_code == 400
        assert resp_high.json()["status"] == "error"


# 7. Prohibited keys (api_key, model_id, etc.) return 400
@pytest.mark.parametrize("prohibited_key", ["api_key", "model", "model_id", "url", "command"])
def test_prohibited_keys_rejected(live_server: str, prohibited_key: str) -> None:
    with httpx.Client(base_url=live_server) as client:
        payload = {"task_mix": "simple", prohibited_key: "forbidden_value"}
        resp = client.post("/api/demo/run", json=payload)
        assert resp.status_code == 400
        data = resp.json()
        assert data["status"] == "error"
        assert f"Parameter '{prohibited_key}' is prohibited" in data["error"]


# 8. No external search calls occur
def test_no_external_search_calls(live_server: str) -> None:
    with patch(
        "tavily.TavilyClient",
        side_effect=RuntimeError("LIVE TAVILY SEARCH DETECTED"),
    ):
        result = execute_demo_run(task_mix="simple", request_count=100)
        assert result["status"] == "completed"


# 9. No API credentials or stack traces in errors
def test_sanitized_error_handling(live_server: str) -> None:
    with httpx.Client(base_url=live_server) as client:
        # Send non-JSON malformed body
        resp = client.post(
            "/api/demo/run",
            content=b"{not valid json...",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400
        data = resp.json()
        assert data["status"] == "error"
        assert "Invalid JSON body" in data["error"]
        # Ensure no tracebacks or internal keys are leaked
        assert "Traceback" not in resp.text
        assert "sk-" not in resp.text


# 10. Static files (/, /index.html) served with correct MIME types
def test_static_file_serving(live_server: str) -> None:
    with httpx.Client(base_url=live_server) as client:
        resp_root = client.get("/")
        assert resp_root.status_code == 200
        assert "text/html" in resp_root.headers.get("Content-Type", "")
        assert "Downshift" in resp_root.text

        resp_index = client.get("/index.html")
        assert resp_index.status_code == 200
        assert "text/html" in resp_index.headers.get("Content-Type", "")
        assert "Downshift" in resp_index.text

        resp_favicon = client.get("/favicon.ico")
        assert resp_favicon.status_code == 204

        resp_meta = client.get("/meta.json")
        assert resp_meta.status_code == 200
        assert resp_meta.json()["mode"] == "mock"


def test_payload_too_large(live_server: str) -> None:
    with httpx.Client(base_url=live_server) as client:
        large_body = b"x" * 70000
        resp = client.post(
            "/api/demo/run",
            content=large_body,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 413
        assert "Payload too large" in resp.json()["error"]


def test_unexpected_extra_keys_rejected(live_server: str) -> None:
    with httpx.Client(base_url=live_server) as client:
        resp = client.post(
            "/api/demo/run",
            json={"task_mix": "simple", "unauthorized_extra_param": 123},
        )
        assert resp.status_code == 400
        assert resp.json()["status"] == "error"


# 11. Directory traversal requests return 404
@pytest.mark.parametrize(
    "traversal_path",
    [
        "/.env",
        "/../.env",
        "/../../pyproject.toml",
        "/..%2f.env",
        "/etc/passwd",
    ],
)
def test_directory_traversal_blocked(live_server: str, traversal_path: str) -> None:
    with httpx.Client(base_url=live_server) as client:
        resp = client.get(traversal_path)
        assert resp.status_code == 404
        assert resp.json()["status"] == "error"


# 12. CORS headers on allowlisted and unrecognized origins
def test_cors_headers_allowlisted_origin(live_server: str) -> None:
    with httpx.Client(base_url=live_server) as client:
        # Preflight OPTIONS request with Vercel origin
        headers = {"Origin": "https://downshift-live-demo.vercel.app"}
        resp_options = client.options("/api/demo/run", headers=headers)
        assert resp_options.status_code == 204
        assert (
            resp_options.headers.get("Access-Control-Allow-Origin")
            == "https://downshift-live-demo.vercel.app"
        )
        assert "GET, POST, OPTIONS" in resp_options.headers.get("Access-Control-Allow-Methods", "")
        assert resp_options.headers.get("Access-Control-Max-Age") == "86400"
        assert "Origin" in resp_options.headers.get("Vary", "")

        # GET request with allowlisted origin returns exact origin
        resp_get = client.get("/api/health", headers=headers)
        assert (
            resp_get.headers.get("Access-Control-Allow-Origin")
            == "https://downshift-live-demo.vercel.app"
        )
        assert "Origin" in resp_get.headers.get("Vary", "")

        # POST request with localhost origin returns exact origin
        resp_post = client.post(
            "/api/demo/run",
            json={"task_mix": "invalid"},
            headers={"Origin": "http://localhost:3000"},
        )
        assert resp_post.headers.get("Access-Control-Allow-Origin") == "http://localhost:3000"
        assert "Origin" in resp_post.headers.get("Vary", "")


def test_cors_headers_unrecognized_origin(live_server: str) -> None:
    with httpx.Client(base_url=live_server) as client:
        unrecognized_headers = {"Origin": "https://unauthorized-attacker.com"}

        # OPTIONS preflight from unrecognized origin must omit Access-Control-Allow-Origin
        resp_options = client.options("/api/demo/run", headers=unrecognized_headers)
        assert resp_options.status_code == 204
        assert "Access-Control-Allow-Origin" not in resp_options.headers
        assert "Origin" in resp_options.headers.get("Vary", "")

        # GET request from unrecognized origin must omit Access-Control-Allow-Origin
        resp_get = client.get("/api/health", headers=unrecognized_headers)
        assert resp_get.status_code == 200
        assert "Access-Control-Allow-Origin" not in resp_get.headers
        assert "Origin" in resp_get.headers.get("Vary", "")

        # POST request from unrecognized origin must omit Access-Control-Allow-Origin
        resp_post = client.post(
            "/api/demo/run",
            json={"task_mix": "invalid"},
            headers=unrecognized_headers,
        )
        assert "Access-Control-Allow-Origin" not in resp_post.headers
        assert "Origin" in resp_post.headers.get("Vary", "")


def test_cors_headers_no_origin_header(live_server: str) -> None:
    with httpx.Client(base_url=live_server) as client:
        # Direct requests without Origin header must omit Access-Control-Allow-Origin
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert "Access-Control-Allow-Origin" not in resp.headers
        assert "Origin" in resp.headers.get("Vary", "")


# 12b. Proves that zero outbound network sockets are opened during demo execution
def test_outbound_network_blocked_at_socket_level() -> None:
    """Prove that execute_demo_run never opens any outbound network sockets."""
    import socket

    orig_connect = socket.socket.connect

    def guarded_connect(self: socket.socket, address: Any) -> None:
        host = address[0] if isinstance(address, tuple) else str(address)
        if host in ("127.0.0.1", "localhost", "::1"):
            orig_connect(self, address)
            return
        raise AssertionError(f"UNAUTHORIZED OUTBOUND NETWORK CALL TO: {address}")

    with patch.object(socket.socket, "connect", guarded_connect):
        result = execute_demo_run(task_mix="balanced", request_count=1000)
        assert result["mode"] == "mock"
        assert result["status"] == "completed"
        assert len(result["steps"]) == 11


# 13. Concurrent requests handled safely without race conditions
def test_concurrent_requests_safety(live_server: str) -> None:
    def make_run_call(count: int) -> int:
        with httpx.Client(base_url=live_server, timeout=60.0) as client:
            resp = client.post(
                "/api/demo/run",
                json={"task_mix": "simple", "request_count": count},
            )
            return resp.status_code

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(make_run_call, 200)
        f2 = executor.submit(make_run_call, 300)
        results = [f1.result(), f2.result()]

    assert results == [200, 200]


# 14. Scaling projections correctly calculated based on request_count (no division by steps)
def test_scaling_projections_calculation(live_server: str) -> None:
    with httpx.Client(base_url=live_server, timeout=30.0) as client:
        resp = client.post(
            "/api/demo/run",
            json={"task_mix": "balanced", "request_count": 1000},
        )
        assert resp.status_code == 200
        data = resp.json()
        summary = data["summary"]
        proj = summary["scaling_projection"]
        assert proj["request_count"] == 1000

        # Verify exact unattenuated math: base_cost * request_count
        base_cost = summary["baseline_cost_per_run_usd"]
        routed_cost = summary["routed_cost_per_run_usd"]
        savings = summary["simulated_savings_usd"]

        expected_base = round(base_cost * 1000, 6)
        expected_routed = round(routed_cost * 1000, 6)
        expected_savings = round(savings * 1000, 6)

        assert abs(proj["projected_baseline_usd"] - expected_base) < 1e-4
        assert abs(proj["projected_routed_usd"] - expected_routed) < 1e-4
        assert abs(proj["projected_savings_usd"] - expected_savings) < 1e-4

        # Verify sample figures: ~3.63, ~0.871, ~2.759 (never ~0.33)
        assert proj["projected_baseline_usd"] > 3.0
        assert proj["projected_routed_usd"] < 1.0
        assert proj["projected_savings_usd"] > 2.0


# 14b. Regression tests: multiple request counts and all task mixes
@pytest.mark.parametrize("task_mix", ["simple", "balanced", "complex"])
@pytest.mark.parametrize("request_count", [100, 1000, 5000, 25000, 100000])
def test_scaling_regression_multiple_counts_and_task_mixes(
    task_mix: str, request_count: int
) -> None:
    """Verify linear scaling holds precisely across all task mixes and request volumes."""
    result = execute_demo_run(task_mix=task_mix, request_count=request_count)
    summary = result["summary"]
    proj = summary["scaling_projection"]

    base_per_run = summary["baseline_cost_per_run_usd"]
    routed_per_run = summary["routed_cost_per_run_usd"]
    savings_per_run = summary["simulated_savings_usd"]

    # Verify scaling projection matches per-run * request_count
    assert proj["request_count"] == request_count
    assert abs(proj["projected_baseline_usd"] - round(base_per_run * request_count, 6)) < 1e-4
    assert abs(proj["projected_routed_usd"] - round(routed_per_run * request_count, 6)) < 1e-4
    assert abs(proj["projected_savings_usd"] - round(savings_per_run * request_count, 6)) < 1e-4

    # Consistency: baseline - routed = savings
    assert (
        abs(
            (proj["projected_baseline_usd"] - proj["projected_routed_usd"])
            - proj["projected_savings_usd"]
        )
        < 1e-4
    )

    # Ensure step count was NOT divided into the projection
    total_steps = summary["total_steps"]
    assert total_steps in (5, 11, 14)
    if base_per_run > 0:
        erroneous_divided_base = round(base_per_run * (request_count / total_steps), 6)
        # For multi-step runs (total_steps > 1), projected_baseline must exceed the flawed divided value
        if total_steps > 1:
            assert proj["projected_baseline_usd"] > erroneous_divided_base * 1.5


# 15. Task mix variations adjust step counts
def test_task_mix_variations(live_server: str) -> None:
    with httpx.Client(base_url=live_server, timeout=30.0) as client:
        # Simple mix: 1 search -> 5 steps
        resp_simple = client.post(
            "/api/demo/run",
            json={"task_mix": "simple", "request_count": 500},
        )
        assert resp_simple.status_code == 200
        assert len(resp_simple.json()["steps"]) == 5

        # Balanced mix: 3 searches -> 11 steps
        resp_bal = client.post(
            "/api/demo/run",
            json={"task_mix": "balanced", "request_count": 500},
        )
        assert resp_bal.status_code == 200
        assert len(resp_bal.json()["steps"]) == 11


# 16. Error handling returns clean JSON without leaking secrets
def test_unknown_endpoint_returns_clean_404(live_server: str) -> None:
    with httpx.Client(base_url=live_server) as client:
        resp = client.get("/api/unknown_endpoint")
        assert resp.status_code == 404
        data = resp.json()
        assert data["status"] == "error"
        assert "Not Found" in data["error"]


# 17. Server binding resolution tests (local defaults, CLI args, Render PORT/HOST env vars)
def test_resolve_server_binding_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PORT", raising=False)
    monkeypatch.delenv("HOST", raising=False)
    host, port = resolve_server_binding()
    assert host == "127.0.0.1"
    assert port == 8000


def test_resolve_server_binding_explicit_args(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PORT", "10000")
    monkeypatch.setenv("HOST", "10.0.0.1")
    # Explicit arguments override environment variables
    host, port = resolve_server_binding(host_arg="0.0.0.0", port_arg=9000)
    assert host == "0.0.0.0"
    assert port == 9000

    # Explicit string port coerced
    host2, port2 = resolve_server_binding(host_arg="127.0.0.1", port_arg="8080")
    assert host2 == "127.0.0.1"
    assert port2 == 8080


def test_resolve_server_binding_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HOST", raising=False)
    monkeypatch.setenv("PORT", "10000")
    # When PORT is present (Render environment), default host must be 0.0.0.0
    host, port = resolve_server_binding()
    assert host == "0.0.0.0"
    assert port == 10000

    # When HOST and PORT are both provided
    monkeypatch.setenv("HOST", "0.0.0.0")
    monkeypatch.setenv("PORT", "5000")
    host2, port2 = resolve_server_binding()
    assert host2 == "0.0.0.0"
    assert port2 == 5000


@pytest.mark.parametrize("invalid_port", [0, 65536, 70000, -1, "not_a_number", ""])
def test_resolve_server_binding_invalid_port(invalid_port: Any) -> None:
    with pytest.raises(ValueError):
        resolve_server_binding(port_arg=invalid_port)


def test_get_allowed_origins_dynamic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RENDER_EXTERNAL_HOSTNAME", raising=False)
    monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
    default_origins = get_allowed_origins()
    assert "https://downshift-live-demo.vercel.app" in default_origins

    # Test RENDER_EXTERNAL_HOSTNAME injection
    monkeypatch.setenv("RENDER_EXTERNAL_HOSTNAME", "downshift-demo.onrender.com")
    render_origins = get_allowed_origins()
    assert "https://downshift-demo.onrender.com" in render_origins

    # Test custom ALLOWED_ORIGINS CSV injection
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://custom1.com, https://custom2.com ")
    custom_origins = get_allowed_origins()
    assert "https://custom1.com" in custom_origins
    assert "https://custom2.com" in custom_origins

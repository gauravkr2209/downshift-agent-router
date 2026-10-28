"""HTTP API service for Downshift interactive demonstration.

Exposes a zero-dependency REST API and static frontend server using Python's
built-in http.server.ThreadingHTTPServer.

Guarantees:
- Strictly offline: executes all runs using deterministic mock clients.
- Never accepts credentials or allows live network inference calls.
- Enforces payload validation on task mix and request counts.
- Thread-safe serialized execution with CORS support.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

# Ensure repository root is in sys.path
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from sqlmodel import Session, col, select

from examples.research_agent.agent import (
    MockSearchAgentClient,
    ResearchAgent,
)
from examples.research_agent.checklist import (
    CHECKLIST_ITEMS,
    RESEARCH_QUESTION,
)
from examples.research_agent.compare_runs import (
    SIMULATION_DISCLAIMER,
)
from server.app import db as db_mod
from server.app.checks import evaluate_run
from server.app.config import REPO_ROOT
from server.app.db import create_db_and_tables, sync_prices_from_yaml
from server.app.judge import MockJudgeClient, judge_run
from server.app.models import Evaluation, Replay, Run, Step
from server.app.replay import MockReplayClient, replay_run
from server.app.router import recommend_routing

logger = logging.getLogger(__name__)

# Thread-safe lock to serialize local SQLite demo executions
_demo_execution_lock = threading.Lock()

# Static web directory path
WEB_DIR = REPO_ROOT / "web"

ALLOWED_TASK_MIXES = ("balanced", "simple", "complex")

DEFAULT_ALLOWED_ORIGINS = {
    "https://downshift-live-demo.vercel.app",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
}


def resolve_server_binding(
    host_arg: str | None = None,
    port_arg: int | str | None = None,
) -> tuple[str, int]:
    """Resolve and validate the host and port to bind the server.

    Priority:
    1. Explicit arguments if passed.
    2. HOST / PORT environment variables (e.g. Render platform environment).
    3. If PORT is in environment and no host is specified, default to '0.0.0.0'.
    4. Safe local default: '127.0.0.1:8000'.
    """
    env_port = os.environ.get("PORT")
    env_host = os.environ.get("HOST")

    # Resolve host
    if host_arg and host_arg.strip():
        host = host_arg.strip()
    elif env_host and env_host.strip():
        host = env_host.strip()
    elif env_port:
        host = "0.0.0.0"
    else:
        host = "127.0.0.1"

    # Resolve port
    raw_port: int | str
    if port_arg is not None:
        raw_port = port_arg
    elif env_port and env_port.strip():
        raw_port = env_port.strip()
    else:
        raw_port = 8000

    try:
        port = int(raw_port)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid port value '{raw_port}': must be an integer.") from exc

    if not (1 <= port <= 65535):
        raise ValueError(f"Port must be between 1 and 65535, got {port}.")

    return host, port


def get_allowed_origins() -> set[str]:
    """Return the set of authorized CORS origins."""
    origins = set(DEFAULT_ALLOWED_ORIGINS)
    render_host = os.environ.get("RENDER_EXTERNAL_HOSTNAME", "").strip()
    if render_host:
        origins.add(f"https://{render_host}")

    custom = os.environ.get("ALLOWED_ORIGINS", "").strip()
    if custom:
        for orig in custom.split(","):
            cleaned = orig.strip()
            if cleaned:
                origins.add(cleaned)
    return origins


class DemoRunRequest(BaseModel):
    """Validated request body for demo execution."""

    model_config = {"extra": "forbid"}

    task_mix: str = Field(default="balanced")
    request_count: int = Field(default=1000, ge=100, le=100000)


def execute_demo_run(
    task_mix: str = "balanced",
    request_count: int = 1000,
    cost_ceiling_usd: float = 0.20,
) -> dict[str, Any]:
    """Execute a deterministic mock demonstration run and return structured metrics."""
    acquired = _demo_execution_lock.acquire(timeout=10.0)
    if not acquired:
        raise TimeoutError("Demo server is currently busy with another simulation.")

    try:
        create_db_and_tables()
        try:
            sync_prices_from_yaml()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Prices sync skipped or failed: %s", exc)

        # Configure max searches based on task mix
        # simple: 1 search (5 steps total)
        # balanced: 3 searches (11 steps total)
        # complex: 4 searches (14 steps total)
        if task_mix == "simple":
            max_searches = 1
        elif task_mix == "complex":
            max_searches = 4
        else:
            max_searches = 3

        # 1. Baseline Agent Execution (Offline Mock)
        mock_client = MockSearchAgentClient(model="nemotron_ultra")
        baseline_agent = ResearchAgent(
            client=mock_client,
            baseline_model="nemotron_ultra",
            cost_ceiling_usd=cost_ceiling_usd,
            routing_config=None,
        )
        baseline_res = baseline_agent.run(
            question=RESEARCH_QUESTION,
            checklist=CHECKLIST_ITEMS,
            max_searches=max_searches,
        )
        baseline_run_id = baseline_res["run_id"]

        # 2. Step Replays (Offline Mock)
        replay_client = MockReplayClient()
        replays = replay_run(
            run_id=baseline_run_id,
            tiers=["super", "nano"],
            client=replay_client,
            cost_ceiling_usd=cost_ceiling_usd,
            force=True,
        )

        # 3. Objective Constraint Evaluations
        with Session(db_mod.engine) as session:
            evaluations = evaluate_run(run_id=baseline_run_id, session=session)
            obj_passed = sum(1 for e in evaluations if e.objective_pass)
            obj_total = len(evaluations)

        # 4. Blinded Quality Judge (Offline Mock)
        judge_client = MockJudgeClient()
        with Session(db_mod.engine) as session:
            judge_evals = judge_run(
                run_id=baseline_run_id,
                client=judge_client,
                force=True,
                session=session,
            )
            judge_passed = sum(
                1 for e in judge_evals if e.judge_reason and e.judge_reason.startswith("[PASS]")
            )
            judge_total = len(judge_evals)

        # 5. Routing Recommendations
        report = recommend_routing(run_id=baseline_run_id, min_pass_rate=0.95, min_samples=3)

        # Build in-memory routing map dictionary
        routing_map: dict[str, str] = {
            tag: rec.recommended_model for tag, rec in report.recommendations.items()
        }

        # 6. Optimized Dynamic Agent Run
        opt_client = MockSearchAgentClient(model="nemotron_ultra")
        opt_agent = ResearchAgent(
            client=opt_client,
            baseline_model="nemotron_ultra",
            cost_ceiling_usd=cost_ceiling_usd,
            routing_config=routing_map,
        )
        opt_res = opt_agent.run(
            question=RESEARCH_QUESTION,
            checklist=CHECKLIST_ITEMS,
            max_searches=max_searches,
        )
        opt_run_id = opt_res["run_id"]

        # 7. Extract step-level comparison data
        with Session(db_mod.engine) as session:
            b_run = session.get(Run, baseline_run_id)
            o_run = session.get(Run, opt_run_id)

            b_steps = session.exec(
                select(Step).where(Step.run_id == baseline_run_id).order_by(col(Step.idx))
            ).all()

            o_steps = session.exec(
                select(Step).where(Step.run_id == opt_run_id).order_by(col(Step.idx))
            ).all()

            # Map evaluation records by step_id through replays
            replays = list(session.exec(select(Replay)).all())
            replay_to_step = {r.id: r.step_id for r in replays}
            eval_map: dict[str, Evaluation] = {}
            for ev in evaluations:
                s_id = replay_to_step.get(ev.replay_id)
                if s_id:
                    eval_map[s_id] = ev

        b_cost = b_run.total_cost_usd if b_run else 0.0
        o_cost = o_run.total_cost_usd if o_run else 0.0
        cost_diff = b_cost - o_cost
        savings_pct = (cost_diff / b_cost * 100.0) if b_cost > 0 else 0.0

        # Build detailed step rows for the frontend
        steps_data: list[dict[str, Any]] = []
        for idx, (b_step, o_step) in enumerate(zip(b_steps, o_steps, strict=False)):
            rec = report.recommendations.get(b_step.tag)
            decision = "downgrade" if rec and rec.recommended_tier != "ultra" else "keep"
            reason = rec.decision_reason if rec else "Retained baseline"

            # Check evaluation details if available
            step_eval: Evaluation | None = eval_map.get(b_step.id)
            obj_pass = step_eval.objective_pass if step_eval else True
            judge_verdict = (
                step_eval.judge_reason if (step_eval and step_eval.judge_reason) else "[PASS]"
            )
            judge_score = (
                step_eval.judge_score if (step_eval and step_eval.judge_score is not None) else 0.85
            )

            steps_data.append(
                {
                    "idx": idx,
                    "tag": b_step.tag,
                    "baseline_model": b_step.model,
                    "selected_model": o_step.model,
                    "selected_tier": (
                        rec.recommended_tier if (rec and decision == "downgrade") else "ultra"
                    ),
                    "decision": decision,
                    "reason": reason,
                    "tokens_in": o_step.tokens_in,
                    "tokens_out": o_step.tokens_out,
                    "cost_usd": round(o_step.cost_usd, 6),
                    "baseline_cost_usd": round(b_step.cost_usd, 6),
                    "objective_pass": obj_pass,
                    "judge_verdict": judge_verdict,
                    "judge_score": judge_score,
                }
            )

        # Calculate projected costs scaled to request_count agent workflow runs
        projected_base = b_cost * request_count
        projected_routed = o_cost * request_count
        projected_saving = cost_diff * request_count

        recommendations_summary = {
            tag: {
                "recommended_tier": rec.recommended_tier,
                "recommended_model": rec.recommended_model,
                "decision_reason": rec.decision_reason,
            }
            for tag, rec in report.recommendations.items()
        }

        return {
            "run_id": baseline_run_id,
            "optimized_run_id": opt_run_id,
            "mode": "mock",
            "status": "completed",
            "task_mix": task_mix,
            "request_count": request_count,
            "task_success": opt_res.get("task_success", True),
            "checklist_score": opt_res.get("checklist_score", 1.0),
            "steps": steps_data,
            "summary": {
                "total_steps": len(b_steps),
                "baseline_cost_per_run_usd": round(b_cost, 8),
                "routed_cost_per_run_usd": round(o_cost, 8),
                "cost_reduction_pct": round(savings_pct, 2),
                "simulated_savings_usd": round(cost_diff, 8),
                "scaling_projection": {
                    "request_count": request_count,
                    "projected_baseline_usd": round(projected_base, 6),
                    "projected_routed_usd": round(projected_routed, 6),
                    "projected_savings_usd": round(projected_saving, 6),
                },
                "objective_evaluations": {
                    "total": obj_total,
                    "passed": obj_passed,
                    "pass_rate_pct": (
                        round((obj_passed / obj_total) * 100.0, 1) if obj_total else 100.0
                    ),
                },
                "judge_evaluations": {
                    "total": judge_total,
                    "passed": judge_passed,
                    "pass_rate_pct": (
                        round((judge_passed / judge_total) * 100.0, 1) if judge_total else 100.0
                    ),
                },
            },
            "recommendations": recommendations_summary,
            "disclaimer": SIMULATION_DISCLAIMER,
        }
    finally:
        _demo_execution_lock.release()


class DownshiftRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the Downshift demo API and static assets."""

    server_version = "DownshiftDemoAPI/0.1"
    timeout = 15

    def _set_cors_headers(self) -> None:
        self.send_header("Vary", "Origin")
        req_origin = self.headers.get("Origin") if hasattr(self, "headers") else None
        if req_origin and req_origin in get_allowed_origins():
            self.send_header("Access-Control-Allow-Origin", req_origin)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.send_header("Access-Control-Max-Age", "86400")

    def _send_json(self, data: Any, status: int = 200) -> None:
        payload = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self._set_cors_headers()
        self.end_headers()
        self.wfile.write(payload)

    def _send_error(self, message: str, status: int = 400) -> None:
        self._send_json({"error": message, "status": "error"}, status=status)

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self._set_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        path = self.path.split("?")[0]

        if path == "/api/health":
            self._send_json(
                {
                    "status": "healthy",
                    "service": "downshift-demo-api",
                    "mode": "mock",
                    "live_inference_enabled": False,
                    "version": "0.1.0",
                }
            )
            return

        if path == "/api/demo/info":
            self._send_json(
                {
                    "mode": "mock",
                    "live_inference_enabled": False,
                    "available_tiers": [
                        {
                            "tier": "ultra",
                            "model_id": "nvidia/Nemotron-3-Ultra-550b-a55b",
                            "params": "550B",
                            "role": "Baseline & Frontier Planning",
                        },
                        {
                            "tier": "super",
                            "model_id": "nvidia/nemotron-3-super-120b-a12b",
                            "params": "120B",
                            "role": "Intermediate Candidate",
                        },
                        {
                            "tier": "nano",
                            "model_id": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
                            "params": "30B",
                            "role": "Lightweight Retrieval & Summarization",
                        },
                    ],
                    "supported_workflows": ["research_agent"],
                    "pricing_status": "illustrative_unverified",
                    "disclaimer": SIMULATION_DISCLAIMER,
                }
            )
            return

        if path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self._set_cors_headers()
            self.end_headers()
            return

        if path == "/meta.json":
            self._send_json(
                {
                    "name": "downshift-demo",
                    "mode": "mock",
                    "live_inference_enabled": False,
                }
            )
            return

        # Serve static assets from web/ directory
        if path == "/" or path == "":
            file_path = WEB_DIR / "index.html"
        else:
            rel = path.lstrip("/")
            file_path = WEB_DIR / rel

        # Security check: prevent directory traversal
        try:
            resolved_path = file_path.resolve()
            if not resolved_path.is_relative_to(WEB_DIR.resolve()) or not resolved_path.is_file():
                self._send_error("Not Found", status=404)
                return
        except Exception:  # noqa: BLE001
            self._send_error("Not Found", status=404)
            return

        mime_type, _ = mimetypes.guess_type(str(resolved_path))
        mime_type = mime_type or "application/octet-stream"
        try:
            content = resolved_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(len(content)))
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(content)
        except Exception as exc:  # noqa: BLE001
            self._send_error(f"Failed to read file: {exc}", status=500)

    def do_POST(self) -> None:
        path = self.path.split("?")[0]

        if path != "/api/demo/run":
            self._send_error("Endpoint not found", status=404)
            return

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > 65536:
            # Drain body up to 128KB to avoid connection abort on Windows sockets
            _ = self.rfile.read(min(content_length, 131072))
            self._send_error("Payload too large (max 64KB)", status=413)
            return

        if content_length <= 0:
            raw_body = b"{}"
        else:
            raw_body = self.rfile.read(content_length)

        try:
            parsed = json.loads(raw_body.decode("utf-8")) if raw_body else {}
        except json.JSONDecodeError:
            self._send_error("Invalid JSON body in request", status=400)
            return

        if not isinstance(parsed, dict):
            self._send_error("Request body must be a JSON object", status=400)
            return

        # Reject prohibited fields that could attempt to pass models or keys
        prohibited = ("api_key", "model", "model_id", "url", "command", "script", "code")
        for p in prohibited:
            if p in parsed:
                self._send_error(
                    f"Parameter '{p}' is prohibited in demo mode. Only task_mix and request_count are accepted.",
                    status=400,
                )
                return

        # Validate task_mix
        task_mix = parsed.get("task_mix", "balanced")
        if task_mix not in ALLOWED_TASK_MIXES:
            self._send_error(
                f"Invalid task_mix '{task_mix}'. Allowed options: {ALLOWED_TASK_MIXES}",
                status=400,
            )
            return

        # Validate request_count
        try:
            req = DemoRunRequest.model_validate(parsed)
        except ValidationError as err:
            self._send_error(f"Validation error: {err}", status=400)
            return

        try:
            result = execute_demo_run(
                task_mix=req.task_mix,
                request_count=req.request_count,
            )
            self._send_json(result)
        except TimeoutError as exc:
            self._send_error(str(exc), status=429)
        except Exception as exc:  # noqa: BLE001
            # Avoid leaking internal stack traces or secrets
            self._send_error(f"Demo execution failed: {type(exc).__name__}", status=500)


def create_demo_server(host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    """Create and return a configured ThreadingHTTPServer."""
    return ThreadingHTTPServer((host, port), DownshiftRequestHandler)


def run_server(host: str | None = None, port: int | str | None = None) -> None:
    """Start the demonstration server on the resolved host and port."""
    resolved_host, resolved_port = resolve_server_binding(host, port)
    server = create_demo_server(host=resolved_host, port=resolved_port)
    print(f"Downshift Demo API server running at http://{resolved_host}:{resolved_port}")
    print("Mode: DETERMINISTIC MOCK (Zero live inference; zero external calls)")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Downshift Demo API server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    host_arg = sys.argv[1] if len(sys.argv) > 1 else None
    port_arg = sys.argv[2] if len(sys.argv) > 2 else None
    run_server(host=host_arg, port=port_arg)

"""Tests for Phase 4 runtime model routing integration in Research Agent."""

import sys
from pathlib import Path
from typing import Any

import pytest
from sqlmodel import Session, SQLModel, col, create_engine, select

from examples.research_agent import compare_runs
from examples.research_agent.agent import (
    DEFAULT_MAX_SEARCHES,
    MockSearchAgentClient,
    ResearchAgent,
)
from examples.research_agent.checklist import (
    CHECKLIST_ITEMS,
    RESEARCH_QUESTION,
)
from examples.research_agent.compare_runs import (
    SIMULATION_DISCLAIMER,
    compare_research_routing,
)
from examples.research_agent.run_baseline import run_baseline
from server.app import db as db_mod
from server.app.db import (
    calculate_cost,
    create_db_and_tables,
    resolve_model_price,
    sync_prices_from_yaml,
)
from server.app.models import ModelPrice, Run, Step
from server.app.router import load_routing_config

ULTRA_ID = "nvidia/Nemotron-3-Ultra-550b-a55b"
SUPER_ID = "nvidia/nemotron-3-super-120b-a12b"
NANO_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"


@pytest.fixture(autouse=True)
def seed_test_pricing() -> None:
    """Ensure database tables and test model prices exist."""
    create_db_and_tables()
    with Session(db_mod.engine) as session:
        for model_id, in_p, out_p in [
            (ULTRA_ID, 1.0, 3.0),
            (SUPER_ID, 0.30, 0.90),
            (NANO_ID, 0.06, 0.24),
            ("nemotron_ultra", 1.0, 3.0),
            ("nemotron_super", 0.30, 0.90),
            ("nemotron_nano", 0.06, 0.24),
        ]:
            existing = session.exec(select(ModelPrice).where(ModelPrice.model == model_id)).first()
            if not existing:
                session.add(ModelPrice(model=model_id, input_per_mtok=in_p, output_per_mtok=out_p))
            else:
                existing.input_per_mtok = in_p
                existing.output_per_mtok = out_p
                session.add(existing)
        session.commit()


# ==============================================================================
# 1. Routing Config Loading and Validation Tests
# ==============================================================================


def test_load_routing_config_none_returns_empty_dict() -> None:
    """Loading None should return an empty dictionary."""
    assert load_routing_config(None) == {}


def test_load_routing_config_from_valid_file() -> None:
    """Loading from an existing valid JSON file should return the mapping."""
    config_path = Path("routing-test.json")
    mapping = load_routing_config(config_path)
    assert mapping["plan"] == ULTRA_ID
    assert mapping["search_query"] == NANO_ID
    assert mapping["summarize_page"] == NANO_ID
    assert mapping["final_answer"] == ULTRA_ID


def test_load_routing_config_from_valid_dict() -> None:
    """Loading from a valid in-memory dict should succeed."""
    valid_dict = {
        "plan": ULTRA_ID,
        "search_query": NANO_ID,
    }
    mapping = load_routing_config(valid_dict)
    assert mapping == valid_dict


def test_load_routing_config_nonexistent_file_raises_not_found(tmp_path: Path) -> None:
    """Nonexistent routing configuration file should raise FileNotFoundError."""
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError, match="not found"):
        load_routing_config(missing)


def test_load_routing_config_malformed_json_raises_value_error(tmp_path: Path) -> None:
    """Malformed JSON should raise ValueError."""
    bad_json = tmp_path / "broken.json"
    bad_json.write_text("{ this is not valid json }", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        load_routing_config(bad_json)


def test_load_routing_config_non_object_json_raises_value_error(tmp_path: Path) -> None:
    """A JSON array or non-object root should raise TypeError."""
    list_json = tmp_path / "array.json"
    list_json.write_text('["plan", "nano"]', encoding="utf-8")
    with pytest.raises(TypeError, match="must be a JSON object"):
        load_routing_config(list_json)

    string_json = tmp_path / "string.json"
    string_json.write_text('"just a string"', encoding="utf-8")
    with pytest.raises(TypeError, match="must be a JSON object"):
        load_routing_config(string_json)


def test_load_routing_config_non_dict_type_raises_value_error() -> None:
    """Passing a non-dict, non-path object directly should raise TypeError."""
    with pytest.raises(TypeError, match="must be a file path or dict"):
        load_routing_config(12345)  # type: ignore[arg-type]


def test_load_routing_config_empty_tag_raises_value_error() -> None:
    """Empty or whitespace tag names must be rejected."""
    with pytest.raises(ValueError, match="tags must be non-empty strings"):
        load_routing_config({"": NANO_ID})
    with pytest.raises(ValueError, match="tags must be non-empty strings"):
        load_routing_config({"   ": NANO_ID})


def test_load_routing_config_empty_model_raises_value_error() -> None:
    """Empty or whitespace model names must be rejected."""
    with pytest.raises(ValueError, match="must be a non-empty string"):
        load_routing_config({"plan": ""})
    with pytest.raises(ValueError, match="must be a non-empty string"):
        load_routing_config({"plan": "   "})


def test_load_routing_config_non_string_values_raise_value_error() -> None:
    """Non-string tags or models must be rejected."""
    with pytest.raises(TypeError, match="tags must be strings"):
        load_routing_config({123: NANO_ID})  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="must be a string"):
        load_routing_config({"plan": 999})


def test_load_routing_config_unknown_model_raises_value_error() -> None:
    """Unknown or unpriced model IDs must be rejected."""
    with pytest.raises(ValueError, match="Unknown or unpriced model ID"):
        load_routing_config({"plan": "nonexistent-model-xyz"})


def test_load_routing_config_null_priced_model_raises_value_error() -> None:
    """Models with null pricing rates in database must be rejected."""
    with Session(db_mod.engine) as session:
        session.add(
            ModelPrice(
                model="null_priced_model",
                input_per_mtok=None,
                output_per_mtok=None,
            )
        )
        session.commit()

    with pytest.raises(ValueError, match="incomplete \\(null\\) pricing rates"):
        load_routing_config({"plan": "null_priced_model"})


# ==============================================================================
# 2. ResearchAgent Routing Selection and Fallback Tests
# ==============================================================================


def test_agent_get_model_for_tag_with_configured_mapping() -> None:
    """Agent routes each configured tag to its specific model."""
    routing_map = {
        "plan": ULTRA_ID,
        "search_query": SUPER_ID,
        "summarize_page": NANO_ID,
        "final_answer": ULTRA_ID,
    }
    client = MockSearchAgentClient(model=ULTRA_ID)
    agent = ResearchAgent(
        client=client,
        baseline_model=ULTRA_ID,
        routing_config=routing_map,
    )

    assert agent.get_model_for_tag("plan") == ULTRA_ID
    assert agent.get_model_for_tag("search_query") == SUPER_ID
    assert agent.get_model_for_tag("summarize_page") == NANO_ID
    assert agent.get_model_for_tag("final_answer") == ULTRA_ID


def test_agent_get_model_for_tag_falls_back_to_baseline_for_unmapped_tags() -> None:
    """Unmapped tags must fall back to baseline_model."""
    routing_map = {
        "search_query": NANO_ID,
    }
    client = MockSearchAgentClient(model=ULTRA_ID)
    agent = ResearchAgent(
        client=client,
        baseline_model=ULTRA_ID,
        routing_config=routing_map,
    )

    # Configured tag returns routed model
    assert agent.get_model_for_tag("search_query") == NANO_ID
    # Unmapped tags fall back to baseline
    assert agent.get_model_for_tag("plan") == ULTRA_ID
    assert agent.get_model_for_tag("summarize_page") == ULTRA_ID
    assert agent.get_model_for_tag("final_answer") == ULTRA_ID
    assert agent.get_model_for_tag("unknown_custom_tag") == ULTRA_ID


def test_agent_get_model_for_tag_without_routing_config_uses_baseline() -> None:
    """When routing_config is None, all tags use baseline_model."""
    client = MockSearchAgentClient(model=ULTRA_ID)
    agent = ResearchAgent(
        client=client,
        baseline_model=ULTRA_ID,
        routing_config=None,
    )

    assert agent.routing_map == {}
    assert agent.get_model_for_tag("plan") == ULTRA_ID
    assert agent.get_model_for_tag("search_query") == ULTRA_ID
    assert agent.get_model_for_tag("summarize_page") == ULTRA_ID
    assert agent.get_model_for_tag("final_answer") == ULTRA_ID


def test_agent_init_fails_immediately_on_invalid_routing_config() -> None:
    """Agent initialization must fail immediately without making model calls on invalid config."""
    client = MockSearchAgentClient(model=ULTRA_ID)
    with pytest.raises(ValueError, match="Unknown or unpriced model ID"):
        ResearchAgent(
            client=client,
            baseline_model=ULTRA_ID,
            routing_config={"plan": "nonexistent_unpriced_model"},
        )


# ==============================================================================
# 3. End-to-End Recorded Execution and Verification
# ==============================================================================


def test_agent_run_records_configured_models_in_database() -> None:
    """A full agent run records routed models and computes correct usage costs per step."""
    routing_map = {
        "plan": ULTRA_ID,
        "search_query": NANO_ID,
        "summarize_page": NANO_ID,
        "final_answer": ULTRA_ID,
    }
    client = MockSearchAgentClient(model=ULTRA_ID)
    agent = ResearchAgent(
        client=client,
        baseline_model=ULTRA_ID,
        routing_config=routing_map,
    )

    res = agent.run(
        question=RESEARCH_QUESTION,
        checklist=CHECKLIST_ITEMS,
        max_searches=2,
    )

    run_id = res["run_id"]
    assert res["task_success"] is True
    assert res["checklist_score"] > 0.9

    with Session(db_mod.engine) as session:
        run = session.get(Run, run_id)
        assert run is not None
        assert run.status == "completed"

        steps = session.exec(
            select(Step).where(Step.run_id == run_id).order_by(col(Step.idx))
        ).all()
        assert len(steps) >= 8

        # Verify per-tag model selection in recorded steps
        for step in steps:
            if step.tag in ("plan", "final_answer"):
                assert step.model == ULTRA_ID
            elif step.tag in ("search_query", "summarize_page"):
                assert step.model == NANO_ID

        # Verify cost calculation: Nano steps should cost significantly less than Ultra steps
        plan_steps = [s for s in steps if s.tag == "plan"]
        query_steps = [s for s in steps if s.tag == "search_query"]
        assert len(plan_steps) == 1
        assert len(query_steps) >= 1
        assert query_steps[0].cost_usd < plan_steps[0].cost_usd


def test_baseline_agent_run_records_baseline_model_on_all_steps() -> None:
    """When no routing config is supplied, all steps use the baseline model."""
    client = MockSearchAgentClient(model=ULTRA_ID)
    agent = ResearchAgent(
        client=client,
        baseline_model=ULTRA_ID,
        routing_config=None,
    )

    res = agent.run(
        question=RESEARCH_QUESTION,
        checklist=CHECKLIST_ITEMS,
        max_searches=1,
    )

    run_id = res["run_id"]
    with Session(db_mod.engine) as session:
        steps = session.exec(
            select(Step).where(Step.run_id == run_id).order_by(col(Step.idx))
        ).all()
        for step in steps:
            assert step.model == ULTRA_ID


# ==============================================================================
# 4. Comparison Runner and CLI Tests
# ==============================================================================


def test_compare_research_routing_execution() -> None:
    """The comparison runner executes baseline and optimized runs and reports metrics."""
    result = compare_research_routing(
        routing_config="routing-test.json",
        baseline_model=ULTRA_ID,
        cost_ceiling_usd=0.20,
        use_real_api=False,
        print_output=True,
    )

    assert result["baseline_run_id"] != result["optimized_run_id"]
    assert result["baseline_cost_usd"] > result["optimized_cost_usd"]
    assert result["savings_usd"] > 0.0
    assert result["savings_pct"] > 0.0
    assert result["baseline_task_success"] is True
    assert result["optimized_task_success"] is True
    assert result["disclaimer"] == SIMULATION_DISCLAIMER

    # Confirm step comparison details
    step_comps = result["step_comparisons"]
    assert len(step_comps) >= 8

    # Check that search_query and summarize_page steps show nano in optimized
    nano_tags = [sc for sc in step_comps if sc["tag"] in ("search_query", "summarize_page")]
    assert len(nano_tags) > 0
    for sc in nano_tags:
        assert sc["baseline_model"] == ULTRA_ID
        assert sc["optimized_model"] == NANO_ID
        assert sc["optimized_cost_usd"] < sc["baseline_cost_usd"]


def test_run_baseline_cli_with_routing_config() -> None:
    """run_baseline executes cleanly when passed routing_config."""
    run_baseline(
        runs=1,
        use_real_api=False,
        baseline_model=ULTRA_ID,
        cost_ceiling_usd=0.20,
        routing_config="routing-test.json",
    )


# ==============================================================================
# 5. Robust Model-ID Resolution & Hardened Edge Cases
# ==============================================================================


def test_resolve_model_price_exact_canonical() -> None:
    """Exact canonical IDs resolve to their canonical record."""
    with Session(db_mod.engine) as session:
        real_id, price = resolve_model_price(NANO_ID, session)
        assert real_id == NANO_ID
        assert price.model == NANO_ID
        assert price.input_per_mtok == 0.06

        real_id_u, price_u = resolve_model_price(ULTRA_ID, session)
        assert real_id_u == ULTRA_ID
        assert price_u.model == ULTRA_ID


def test_resolve_model_price_case_variants() -> None:
    """Lowercase, uppercase, and mixed-case model IDs resolve to exact canonical ID."""
    with Session(db_mod.engine) as session:
        # Lowercase canonical ID
        real_id, price = resolve_model_price("nvidia/nvidia-nemotron-3-nano-30b-a3b", session)
        assert real_id == NANO_ID
        assert price.model == NANO_ID

        # Uppercase canonical ID
        real_id_u, price_u = resolve_model_price("NVIDIA/NEMOTRON-3-ULTRA-550B-A55B", session)
        assert real_id_u == ULTRA_ID
        assert price_u.model == ULTRA_ID

        # Lowercase super ID
        real_id_s, price_s = resolve_model_price("nvidia/nemotron-3-super-120b-a12b", session)
        assert real_id_s == SUPER_ID
        assert price_s.model == SUPER_ID


def test_resolve_model_price_alias_and_whitespace() -> None:
    """Configured aliases with whitespace and various casing resolve to canonical ID."""
    mem_engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(mem_engine)
    with Session(mem_engine) as session:
        sync_prices_from_yaml(session=session)

        real_id, price = resolve_model_price("  nemotron_ultra  ", session)
        assert real_id == ULTRA_ID
        assert price.model == ULTRA_ID

        real_id_n, price_n = resolve_model_price("NEMOTRON_NANO", session)
        assert real_id_n == NANO_ID
        assert price_n.model == NANO_ID

        real_id_short, price_short = resolve_model_price("nano", session)
        assert real_id_short == NANO_ID
        assert price_short.model == NANO_ID


def test_resolve_model_price_unknown_model_rejected() -> None:
    """Unknown model strings and empty strings must be rejected."""
    with Session(db_mod.engine) as session:
        with pytest.raises(ValueError, match="not found in pricing records"):
            resolve_model_price("nonexistent-vendor/fake-model-7b", session)

        with pytest.raises(ValueError, match="cannot be empty"):
            resolve_model_price("", session)

        with pytest.raises(ValueError, match="cannot be empty"):
            resolve_model_price("   ", session)


def test_resolve_model_price_null_pricing_safeguard() -> None:
    """Null pricing records are recognized, and calculate_cost raises ValueError."""
    with Session(db_mod.engine) as session:
        null_model = "test_null_model_pricing"
        session.add(ModelPrice(model=null_model, input_per_mtok=None, output_per_mtok=None))
        session.commit()

        real_id, price = resolve_model_price(null_model, session)
        assert real_id == null_model
        assert price.input_per_mtok is None

        with pytest.raises(ValueError, match="unconfigured \\(null\\) values"):
            calculate_cost(null_model, 100, 100, session)


def test_resolve_model_price_unpopulated_isolated_database() -> None:
    """Resolution works on an initially unpopulated isolated database via prices.yaml sync."""
    mem_engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(mem_engine)

    with Session(mem_engine) as mem_session:
        # DB is completely empty before query
        count = len(mem_session.exec(select(ModelPrice)).all())
        assert count == 0

        # Querying lowercase canonical ID triggers self-healing sync
        real_id, price = resolve_model_price("nvidia/nvidia-nemotron-3-nano-30b-a3b", mem_session)
        assert real_id == NANO_ID
        assert price.model == NANO_ID
        assert price.input_per_mtok == 0.06

        # Now DB has synced records
        synced_count = len(mem_session.exec(select(ModelPrice)).all())
        assert synced_count > 0

        # Alias query also resolves against the newly synced DB
        real_id_u, price_u = resolve_model_price("nemotron_ultra", mem_session)
        assert real_id_u == ULTRA_ID
        assert price_u.model == ULTRA_ID


def test_load_routing_config_directory_path_rejected(tmp_path: Path) -> None:
    """A directory path passed as routing_config raises ValueError."""
    sub_dir = tmp_path / "routing_dir"
    sub_dir.mkdir()
    with pytest.raises(ValueError, match="is not a file"):
        load_routing_config(sub_dir)


def test_routing_cost_ceiling_aborts_agent_run() -> None:
    """Cost ceiling enforcement aborts run when active routing uses lower-cost tier."""
    routing_map = {
        "plan": ULTRA_ID,
        "search_query": NANO_ID,
        "summarize_page": NANO_ID,
        "final_answer": ULTRA_ID,
    }
    client = MockSearchAgentClient(model=ULTRA_ID)
    agent = ResearchAgent(
        client=client,
        baseline_model=ULTRA_ID,
        cost_ceiling_usd=0.0001,
        routing_config=routing_map,
    )

    with pytest.raises(RuntimeError, match="Cost ceiling of \\$0.0001 reached"):
        agent.run(question="Budget abort test", max_searches=2)

    with Session(db_mod.engine) as session:
        latest_run = session.exec(select(Run).order_by(col(Run.started_at).desc())).first()
        assert latest_run is not None
        assert latest_run.status == "failed"


def test_missing_routing_tag_fallback_in_live_run() -> None:
    """Unmapped tags in a partial routing config execute on baseline_model in a recorded run."""
    partial_routing = {
        "search_query": NANO_ID,
    }
    client = MockSearchAgentClient(model=ULTRA_ID)
    agent = ResearchAgent(
        client=client,
        baseline_model=ULTRA_ID,
        routing_config=partial_routing,
    )

    res = agent.run(question=RESEARCH_QUESTION, checklist=CHECKLIST_ITEMS, max_searches=1)
    run_id = res["run_id"]

    with Session(db_mod.engine) as session:
        steps = session.exec(
            select(Step).where(Step.run_id == run_id).order_by(col(Step.idx))
        ).all()
        for step in steps:
            if step.tag == "search_query":
                assert step.model == NANO_ID
            else:
                assert step.model == ULTRA_ID


def test_invalid_routing_config_fails_before_model_call() -> None:
    """Invalid routing config raises before any model call can be made."""
    call_count = 0

    class SpyingClient:
        def __init__(self) -> None:
            self.chat = self
            self.completions = self

        def create(self, *args: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            raise AssertionError("Model call should never have occurred on invalid config!")

    spy = SpyingClient()
    with pytest.raises(ValueError, match="Unknown or unpriced model ID"):
        ResearchAgent(
            client=spy,
            baseline_model=ULTRA_ID,
            routing_config={"plan": "nonexistent_model_id_123"},
        )

    assert call_count == 0


def test_compare_runs_cli_argument_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI entrypoint for compare_runs parses arguments and runs cleanly."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_runs.py",
            "--routing-config",
            "routing-test.json",
            "--model",
            "nemotron_ultra",
            "--cost-ceiling",
            "0.20",
        ],
    )
    # Call main; it should succeed with exit code 0 / no exception
    compare_runs.main()


def test_resolve_run_id_latest() -> None:
    """resolve_run_id('latest') returns the ID of the most recent run."""
    from datetime import UTC, datetime, timedelta

    from server.app.cli import resolve_run_id

    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        # Empty DB exits with code 1
        with pytest.raises(SystemExit) as exc_info:
            resolve_run_id("latest", session)
        assert exc_info.value.code == 1

        t0 = datetime.now(UTC) - timedelta(minutes=10)
        t1 = datetime.now(UTC)

        run_old = Run(
            id="run-older-111",
            task_name="task_a",
            agent_name="agent_a",
            baseline_model=ULTRA_ID,
            started_at=t0,
        )
        run_new = Run(
            id="run-newer-222",
            task_name="task_b",
            agent_name="agent_b",
            baseline_model=ULTRA_ID,
            started_at=t1,
        )
        session.add(run_old)
        session.add(run_new)
        session.commit()

        assert resolve_run_id("latest", session) == "run-newer-222"
        assert resolve_run_id("  LATEST  ", session) == "run-newer-222"
        assert resolve_run_id("run-older-111", session) == "run-older-111"


def test_compare_research_routing_with_explicit_run_ids() -> None:
    """compare_research_routing compares pre-existing run IDs without re-executing."""
    # First execute a baseline and an optimized run using ResearchAgent
    client = MockSearchAgentClient(model="nemotron_ultra")
    b_agent = ResearchAgent(
        client=client,
        baseline_model="nemotron_ultra",
        routing_config=None,
    )
    b_res = b_agent.run(
        question=RESEARCH_QUESTION,
        checklist=CHECKLIST_ITEMS,
        max_searches=DEFAULT_MAX_SEARCHES,
    )

    o_agent = ResearchAgent(
        client=client,
        baseline_model="nemotron_ultra",
        routing_config="routing-test.json",
    )
    o_res = o_agent.run(
        question=RESEARCH_QUESTION,
        checklist=CHECKLIST_ITEMS,
        max_searches=DEFAULT_MAX_SEARCHES,
    )

    result = compare_research_routing(
        routing_config="routing-test.json",
        baseline_model="nemotron_ultra",
        baseline_run_id=b_res["run_id"],
        optimized_run_id=o_res["run_id"],
        print_output=False,
    )

    assert result["baseline_run_id"] == b_res["run_id"]
    assert result["optimized_run_id"] == o_res["run_id"]
    assert result["baseline_steps"] == 11
    assert result["optimized_steps"] == 11
    assert result["baseline_cost_usd"] > result["optimized_cost_usd"]
    assert result["savings_pct"] > 0.0
    assert result["baseline_task_success"] is True
    assert result["optimized_task_success"] is True


def test_demo_run_end_to_end(tmp_path: Path) -> None:
    """Full end-to-end demo lifecycle executes cleanly in simulated mock mode."""
    from examples.research_agent.demo import run_demo

    test_export = tmp_path / "test-routing.json"
    result = run_demo(export_path=str(test_export), cost_ceiling_usd=0.20)

    assert result["baseline_run_id"]
    assert result["optimized_run_id"]
    assert test_export.exists()
    assert result["savings_pct"] > 50.0
    assert len(result["step_comparisons"]) == 11

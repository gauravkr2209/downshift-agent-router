"""Tests for Research Agent execution, diagnostics, history sanitization, and checklist scoring."""

import logging
import re
from pathlib import Path
from typing import Any

import pytest
from sqlmodel import Session, col, create_engine, select

from examples.research_agent.agent import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    FINAL_ANSWER_MAX_TOKENS,
    PLAN_MAX_TOKENS,
    TOOL_RESULT_MAX_CHARS,
    MockSearchAgentClient,
    ResearchAgent,
    _MockChoice,
    _MockMessage,
    _MockResponse,
    _MockUsage,
    sanitize_history_message,
    truncate_tool_result,
)
from examples.research_agent.checklist import (
    RESEARCH_QUESTION,
    evaluate_checklist_compliance,
)
from examples.research_agent.run_baseline import run_baseline
from server.app import db as db_mod
from server.app.config import settings
from server.app.models import ModelPrice, Run, Step


@pytest.fixture(autouse=True)
def seed_pricing() -> None:
    """Seed test pricing for nemotron_ultra."""
    with Session(db_mod.engine) as session:
        session.add(
            ModelPrice(
                model="nemotron_ultra",
                input_per_mtok=1.5,
                output_per_mtok=3.0,
            )
        )
        session.commit()


def test_research_agent_multi_search_ten_plus_steps_and_token_limits() -> None:
    """Test that default run executes 2-3 searches with >=10 steps, max_tokens=3000 on plan and final_answer."""
    client = MockSearchAgentClient(model="nemotron_ultra")
    agent = ResearchAgent(
        client=client,
        baseline_model="nemotron_ultra",
        search_engine_name="Mock",
    )

    result = agent.run(
        question="What are the key technical specifications?",
        checklist=[["essential facts"], ["specifications", "specs"]],
        max_searches=3,
    )
    run_id = result["run_id"]
    assert run_id is not None
    assert result["task_success"] is True
    assert result["checklist_score"] == pytest.approx(1.0)
    assert result["search_engine_name"] == "Mock"
    assert len(result["search_stats"]) == 3

    # Inspect steps in database
    with Session(db_mod.engine) as session:
        steps = session.exec(
            select(Step).where(Step.run_id == run_id).order_by(col(Step.idx))
        ).all()

        # Step count: 1 plan + 3 queries + 6 summaries + 1 final answer = 11 steps (>= 10 steps)
        assert len(steps) >= 10
        assert len(steps) == 11

        plan_step = next(s for s in steps if s.tag == "plan")
        final_step = next(s for s in steps if s.tag == "final_answer")
        query_steps = [s for s in steps if s.tag == "search_query"]
        summary_steps = [s for s in steps if s.tag == "summarize_page"]

        # Plan and final_answer must have max_tokens=3000
        assert plan_step.request["max_tokens"] == PLAN_MAX_TOKENS
        assert final_step.request["max_tokens"] == FINAL_ANSWER_MAX_TOKENS
        # Other steps have max_tokens=1500
        assert query_steps[0].request["max_tokens"] == DEFAULT_MAX_TOKENS
        assert summary_steps[0].request["max_tokens"] == DEFAULT_MAX_TOKENS

        # Plan prompt must ask for a short numbered plan only
        plan_user_msg = plan_step.request["messages"][1]["content"]
        assert "short numbered plan only" in plan_user_msg

        # All steps must record temperature=0.2
        for s in steps:
            assert s.request["temperature"] == DEFAULT_TEMPERATURE


def test_finish_reason_reporting_and_length_flag() -> None:
    """Test finish_reason output and flagging when finish_reason=='length'."""

    class LengthMockClient(MockSearchAgentClient):
        def create(self, *args: Any, **kwargs: Any) -> _MockResponse:
            super().create(*args, **kwargs)
            return _MockResponse(
                model=self.model,
                usage=_MockUsage(prompt_tokens=100, completion_tokens=100, total_tokens=200),
                choices=[
                    _MockChoice(
                        message=_MockMessage(content="Truncated output..."),
                        finish_reason="length",
                    )
                ],
            )

    client = LengthMockClient(model="nemotron_ultra")
    agent = ResearchAgent(client=client, baseline_model="nemotron_ultra")
    res = agent.run(question="Test question", max_searches=1)

    with Session(db_mod.engine) as session:
        steps = session.exec(select(Step).where(Step.run_id == res["run_id"])).all()
        # Verify length finish_reason is stored in DB
        assert any(
            s.response and s.response.get("choices", [{}])[0].get("finish_reason") == "length"
            for s in steps
        )


def test_baseline_run_diagnostic_output(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test that run_baseline prints research question, items loaded, answer snippet, search details, and finish_reasons."""
    import examples.research_agent.run_baseline as rb

    monkeypatch.setattr(rb, "RESEARCH_QUESTION", "Diagnostic inquiry about system architecture?")
    monkeypatch.setattr(
        rb,
        "CHECKLIST_ITEMS",
        [["essential facts", "core facts"], ["specifications"]],
    )

    run_baseline(runs=1, use_real_api=False)
    captured = capsys.readouterr().out

    # 1. Start of run output
    assert "Research Question        : Diagnostic inquiry about system architecture?" in captured
    assert "Checklist Items Loaded   : 2" in captured

    # 2. Per step finish_reason
    assert "finish_reason: stop" in captured

    # 3. Search provider and count details
    assert "Search Execution (Mock):" in captured
    assert "returned 2 result(s)" in captured

    # 4. Final answer snippet (first 3000 chars)
    assert "--- Final Answer (first 3000 characters) ---" in captured
    assert "Synthesized final answer" in captured

    # 5. Checklist per-item results
    assert "--- Checklist Results per Item ---" in captured
    assert "[FOUND] essential facts/core facts" in captured
    assert "[FOUND] specifications" in captured


def test_truncate_tool_result_and_logging(caplog: pytest.LogCaptureFixture) -> None:
    """Test that tool results over 2000 chars are truncated to 2000 and logged."""
    with caplog.at_level(logging.INFO):
        long_content = "x" * 2500
        truncated = truncate_tool_result(long_content)
        assert len(truncated) == TOOL_RESULT_MAX_CHARS
        assert "Truncated tool result from 2500 to 2000 characters." in caplog.text

    caplog.clear()
    short_content = "short content"
    assert truncate_tool_result(short_content) == short_content
    assert "Truncated" not in caplog.text


def test_tool_truncation_applied_in_agent_history() -> None:
    """Test that search result truncation is applied before adding to agent conversation history."""
    long_source = "A" * 3000

    def mock_long_search(_query: str) -> list[str]:
        return [long_source]

    client = MockSearchAgentClient(model="nemotron_ultra")
    agent = ResearchAgent(
        client=client,
        baseline_model="nemotron_ultra",
        search_fn=mock_long_search,
    )

    res = agent.run(question="Search test", max_searches=1)
    history = res["conversation_history"]

    doc_msgs = [m for m in history if "Summarize key facts from document:" in m.get("content", "")]
    assert len(doc_msgs) == 1
    prefix = "Summarize key facts from document: "
    doc_content = doc_msgs[0]["content"].removeprefix(prefix)
    assert len(doc_content) == TOOL_RESULT_MAX_CHARS


def test_reasoning_fields_not_sent_back_in_history() -> None:
    """Test that assistant reasoning fields are excluded from forwarded history but preserved in Step.response."""
    dirty_msg = {
        "role": "assistant",
        "content": "Here is the response.",
        "reasoning_content": "Deep internal reasoning that should never be echoed.",
        "thought": "Internal thinking scratchpad.",
        "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "test"}}],
    }
    clean_msg = sanitize_history_message(dirty_msg)
    assert "reasoning_content" not in clean_msg
    assert "thought" not in clean_msg
    assert clean_msg["role"] == "assistant"
    assert clean_msg["content"] == "Here is the response."
    assert "tool_calls" in clean_msg

    client = MockSearchAgentClient(model="nemotron_ultra")
    agent = ResearchAgent(client=client, baseline_model="nemotron_ultra")

    res = agent.run(question="Test question", max_searches=1)
    history = res["conversation_history"]

    for msg in history:
        if msg.get("role") == "assistant":
            assert "reasoning_content" not in msg
            assert "thought" not in msg
            assert "thinking" not in msg

    with Session(db_mod.engine) as session:
        steps = session.exec(select(Step).where(Step.run_id == res["run_id"])).all()
        plan_step = next(s for s in steps if s.tag == "plan")
        assert plan_step.response is not None
        resp_choices = plan_step.response.get("choices", [])
        assert len(resp_choices) > 0
        assistant_msg = resp_choices[0].get("message", {})
        assert assistant_msg.get("reasoning_content") is not None


def test_checklist_keyword_groups_and_fractional_score() -> None:
    """Test keyword group checklist matching, fractional score, and storage in Run.meta."""
    answer = "We evaluated NVIDIA hardware using Nemotron-4 models."
    checklist = [
        ["nvidia", "nebius"],
        ["nemotron-4", "nemotron 4"],
        ["missing_tech", "unavailable_spec"],
    ]
    eval_res = evaluate_checklist_compliance(answer, checklist)
    assert eval_res["passed_count"] == 2
    assert eval_res["total_count"] == 3
    assert eval_res["checklist_score"] == pytest.approx(2.0 / 3.0)
    assert eval_res["all_passed"] is False

    client = MockSearchAgentClient(model="nemotron_ultra")
    agent = ResearchAgent(client=client, baseline_model="nemotron_ultra")

    test_checklist = [
        ["essential facts", "core findings"],
        ["technical specifications", "specs"],
        ["nonexistent_fact_alpha", "beta"],
        ["nonexistent_fact_gamma"],
    ]
    res = agent.run(question="Research query", checklist=test_checklist, max_searches=1)
    assert res["task_success"] is False
    assert res["checklist_score"] == pytest.approx(0.5)

    with Session(db_mod.engine) as session:
        run_record = session.get(Run, res["run_id"])
        assert run_record is not None
        assert run_record.task_success is False
        assert run_record.meta is not None
        assert run_record.meta["checklist_score"] == pytest.approx(0.5)


def test_real_flag_refusal_when_placeholder_or_empty_checklist(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test that --real mode refuses to run if question is a placeholder or checklist is empty."""
    import examples.research_agent.run_baseline as rb

    monkeypatch.setattr(rb, "RESEARCH_QUESTION", "PLACEHOLDER_RESEARCH_QUESTION")
    monkeypatch.setattr(rb, "CHECKLIST_ITEMS", [["item1"]])

    with pytest.raises(SystemExit) as exc1:
        run_baseline(runs=1, use_real_api=True)
    assert exc1.value.code == 1
    assert (
        "REFUSAL: RESEARCH_QUESTION in checklist.py is still a placeholder."
        in capsys.readouterr().err
    )

    monkeypatch.setattr(rb, "RESEARCH_QUESTION", "What is the capital of France?")
    monkeypatch.setattr(rb, "CHECKLIST_ITEMS", [])

    with pytest.raises(SystemExit) as exc2:
        run_baseline(runs=1, use_real_api=True)
    assert exc2.value.code == 1
    assert "REFUSAL: CHECKLIST_ITEMS in checklist.py is empty." in capsys.readouterr().err


def test_research_agent_cost_ceiling_abort() -> None:
    """Test that per-run cost ceiling aborts agent execution before exceeding budget."""
    client = MockSearchAgentClient(model="nemotron_ultra")
    agent = ResearchAgent(
        client=client,
        baseline_model="nemotron_ultra",
        cost_ceiling_usd=0.0001,
    )

    with pytest.raises(RuntimeError, match="Cost ceiling of \\$0.0001 reached"):
        agent.run(question="High cost inquiry")

    with Session(db_mod.engine) as session:
        runs = session.exec(select(Run).order_by(col(Run.started_at).desc())).all()
        assert len(runs) > 0
        latest_run = runs[0]
        assert latest_run.status == "failed"


def test_mock_baseline_syncs_prices_and_succeeds_on_empty_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that run_baseline in mock mode automatically syncs prices.yaml into a fresh database and succeeds."""
    import examples.research_agent.run_baseline as rb

    # 1. Use an isolated temporary SQLite database under tmp_path
    fresh_db_file = tmp_path / "fresh_mock_baseline.db"
    fresh_engine = create_engine(
        f"sqlite:///{fresh_db_file}",
        connect_args={"check_same_thread": False},
    )
    monkeypatch.setattr(settings, "database_url", f"sqlite:///{fresh_db_file}")
    monkeypatch.setattr(db_mod, "engine", fresh_engine)

    # 2. Use a temporary prices.yaml with test-only model ID and non-null prices
    dummy_prices = tmp_path / "prices.yaml"
    dummy_prices.write_text(
        """
models:
  nemotron_ultra:
    model_id: "nvidia/nemotron-ultra-mock"
    input_per_mtok: 1.5
    output_per_mtok: 3.0
"""
    )
    monkeypatch.setattr(settings, "prices_path", dummy_prices)
    monkeypatch.setattr(settings, "ultra_model_id", "nvidia/nemotron-ultra-mock")
    monkeypatch.setattr(rb, "RESEARCH_QUESTION", "Test inquiry about architecture?")
    monkeypatch.setattr(rb, "CHECKLIST_ITEMS", [["essential facts"], ["specifications"]])

    try:
        # 3. Run mock baseline (real API disabled)
        run_baseline(runs=1, use_real_api=False, baseline_model="nvidia/nemotron-ultra-mock")

        # 4. Verify prices synchronized automatically and run succeeded
        with Session(fresh_engine) as session:
            synced_price = session.exec(
                select(ModelPrice).where(ModelPrice.model == "nvidia/nemotron-ultra-mock")
            ).first()
            assert synced_price is not None
            assert synced_price.input_per_mtok == 1.5
            assert synced_price.output_per_mtok == 3.0

            runs = session.exec(select(Run)).all()
            assert len(runs) == 1
            assert runs[0].status == "completed"
            assert runs[0].task_success is True
    finally:
        # 5. Clean up fresh engine reliably even on assertion failure
        fresh_engine.dispose()


def test_mock_search_queries_differ_and_synthesize_results() -> None:
    """Test that mock client produces distinct search queries, synthesizes research, and passes configured checklist."""
    client = MockSearchAgentClient(model="nemotron_ultra")
    agent = ResearchAgent(
        client=client,
        baseline_model="nemotron_ultra",
        search_engine_name="Mock",
    )

    # Run agent using default RESEARCH_QUESTION and CHECKLIST_ITEMS
    result = agent.run(max_searches=3)
    search_stats = result["search_stats"]
    assert len(search_stats) == 3

    # 1. Verify queries are distinct and address separate intents
    queries = [st["query"] for st in search_stats]
    assert len(set(queries)) == 3
    assert any("nebius" in q.lower() or "token factory" in q.lower() for q in queries)
    assert any("tier" in q.lower() or "ultra" in q.lower() or "nano" in q.lower() for q in queries)
    assert any(
        "agent" in q.lower() or "function calling" in q.lower() or "reasoning" in q.lower()
        for q in queries
    )

    # 2. Verify final answer synthesizes mock findings and is marked unverified/simulated
    final_answer = result["final_answer"]
    assert "Synthesized final answer" in final_answer
    assert "[SIMULATED MOCK RESEARCH - UNVERIFIED DATA]" in final_answer
    assert "Key findings from simulated research steps:" in final_answer

    # 3. Verify checklist scoring
    assert result["task_success"] is True
    assert result["checklist_score"] == pytest.approx(1.0)


def test_evaluate_checklist_compliance_word_boundaries_rag_not_storage() -> None:
    """Test that keywords are matched with word boundaries (e.g. 'rag' does not match 'storage')."""
    checklist = [["rag"]]
    # Should not match 'storage'
    storage_res = evaluate_checklist_compliance(
        "We configured persistent cloud storage.", checklist
    )
    assert storage_res["checklist_score"] == 0.0
    assert storage_res["passed_count"] == 0
    assert storage_res["all_passed"] is False

    # Should match standalone 'rag' case-insensitively
    rag_res = evaluate_checklist_compliance("Evaluation of agentic RAG workflows.", checklist)
    assert rag_res["checklist_score"] == 1.0
    assert rag_res["passed_count"] == 1
    assert rag_res["all_passed"] is True


def test_no_checklist_keyword_appears_in_research_question() -> None:
    """Verify evaluation when no checklist keyword appears in RESEARCH_QUESTION and target facts are absent."""
    disjoint_checklist = [["quantum", "supercomputing"], ["biological synthesis"]]
    res = evaluate_checklist_compliance(RESEARCH_QUESTION, disjoint_checklist)
    assert res["checklist_score"] == 0.0
    assert res["passed_count"] == 0
    assert res["all_passed"] is False
    assert len(res["missing_groups"]) == 2

    # Verify that target discovery facts (model tiers, parameter sizes, agent tooling)
    # do not appear in the research question prompt itself:
    discovery_facts = [
        "ultra",
        "super",
        "nano",
        "340b",
        "70b",
        "120b",
        "30b",
        "4b",
        "function calling",
        "tool use",
        "reasoning",
        "rag",
    ]
    for kw in discovery_facts:
        pattern = rf"\b{re.escape(kw)}\b"
        assert not re.search(pattern, RESEARCH_QUESTION, re.IGNORECASE), (
            f"Discovery fact '{kw}' unexpectedly appeared in RESEARCH_QUESTION"
        )


def test_evaluate_checklist_compliance_research_question_scores_below_one() -> None:
    """Verify that evaluate_checklist_compliance(RESEARCH_QUESTION) scores below 1.0."""
    res = evaluate_checklist_compliance(RESEARCH_QUESTION)
    assert res["checklist_score"] < 1.0
    assert res["all_passed"] is False

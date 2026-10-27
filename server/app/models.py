"""Database models for Downshift."""

import uuid
from datetime import UTC, datetime
from typing import Any, Optional

from sqlalchemy import Index
from sqlmodel import JSON, Column, Field, Relationship, SQLModel


def generate_uuid() -> str:
    """Generate a unique identifier string."""
    return str(uuid.uuid4())


class ModelPrice(SQLModel, table=True):
    """Pricing per million tokens for a model tier."""

    __tablename__ = "model_prices"

    model: str = Field(primary_key=True)
    input_per_mtok: float | None = Field(default=None, nullable=True)
    output_per_mtok: float | None = Field(default=None, nullable=True)
    source: str = Field(default="Nebius Token Factory")
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Run(SQLModel, table=True):
    """Recorded agent execution run."""

    __tablename__ = "runs"

    id: str = Field(default_factory=generate_uuid, primary_key=True)
    task_name: str
    agent_name: str
    baseline_model: str
    status: str = Field(default="running")
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    total_cost_usd: float = Field(default=0.0)
    tokens_in: int = Field(default=0)
    tokens_out: int = Field(default=0)
    task_success: bool | None = Field(default=None, nullable=True)
    meta: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON, nullable=True))

    steps: list["Step"] = Relationship(back_populates="run")


class Step(SQLModel, table=True):
    """Single execution step within an agent run."""

    __tablename__ = "steps"
    __table_args__ = (Index("idx_steps_run_idx", "run_id", "idx"),)

    id: str = Field(default_factory=generate_uuid, primary_key=True)
    run_id: str = Field(foreign_key="runs.id", index=True)
    idx: int = Field(index=True)
    tag: str = Field(index=True)
    kind: str  # "tool_call" or "text"
    model: str
    request: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    response: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON, nullable=True))
    tool_calls: list[dict[str, Any]] | None = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )
    tokens_in: int = Field(default=0)
    tokens_out: int = Field(default=0)
    latency_ms: float = Field(default=0.0)
    cost_usd: float = Field(default=0.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    run: Run | None = Relationship(back_populates="steps")
    replays: list["Replay"] = Relationship(back_populates="step")


class Replay(SQLModel, table=True):
    """Replay of a recorded step on a candidate model tier."""

    __tablename__ = "replays"
    __table_args__ = (Index("idx_replays_step_tier", "step_id", "tier"),)

    id: str = Field(default_factory=generate_uuid, primary_key=True)
    step_id: str = Field(foreign_key="steps.id", index=True)
    tier: str
    model: str
    mode: str = Field(default="step_level")
    response: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON, nullable=True))
    tool_calls: list[dict[str, Any]] | None = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )
    tokens_in: int = Field(default=0)
    tokens_out: int = Field(default=0)
    latency_ms: float = Field(default=0.0)
    cost_usd: float = Field(default=0.0)
    error: str | None = Field(default=None, nullable=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    step: Step | None = Relationship(back_populates="replays")
    evaluation: Optional["Evaluation"] = Relationship(
        back_populates="replay", sa_relationship_kwargs={"uselist": False}
    )


class Evaluation(SQLModel, table=True):
    """Objective and LLM judge evaluation of a replayed step."""

    __tablename__ = "evaluations"
    __table_args__ = (Index("idx_evaluations_replay", "replay_id"),)

    id: str = Field(default_factory=generate_uuid, primary_key=True)
    replay_id: str = Field(foreign_key="replays.id", index=True, unique=True)
    objective_pass: bool = Field(default=False)
    objective_detail: dict[str, Any] | None = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )
    judge_score: float | None = Field(default=None, nullable=True)
    judge_reason: str | None = Field(default=None, nullable=True)
    judge_model: str | None = Field(default=None, nullable=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    replay: Replay | None = Relationship(back_populates="evaluation")

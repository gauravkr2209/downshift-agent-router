"""Database engine, session management, and pricing calculations."""

import sqlite3
from collections.abc import Generator
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, col, create_engine, func, or_, select

from server.app.config import REPO_ROOT, settings
from server.app.models import ModelPrice, Run, Step  # noqa: F401

connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, echo=False, connect_args=connect_args)


def configure_sqlite_engine(target_engine: Any) -> None:
    """Enable WAL mode and a busy timeout on an engine for SQLite connections only."""

    @event.listens_for(target_engine, "connect")
    def _set_sqlite_pragma(dbapi_connection: Any, connection_record: Any) -> None:
        if isinstance(dbapi_connection, sqlite3.Connection):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.close()


@event.listens_for(Engine, "connect")
def _set_sqlite_engine_pragmas(dbapi_connection: Any, connection_record: Any) -> None:
    if isinstance(dbapi_connection, sqlite3.Connection):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()


configure_sqlite_engine(engine)


def create_db_and_tables() -> None:
    """Create all configured database tables."""
    SQLModel.metadata.create_all(engine)


def get_session() -> Generator[Session, None, None]:
    """Yield a database session context."""
    with Session(engine) as session:
        yield session


def sync_prices_from_yaml(
    yaml_path: str | Path | None = None,
    session: Session | None = None,
) -> None:
    """Sync model pricing from prices.yaml into the database."""
    create_db_and_tables()
    target_path = Path(yaml_path) if yaml_path else settings.prices_path
    if not target_path.is_absolute():
        target_path = REPO_ROOT / target_path

    if not target_path.exists():
        raise FileNotFoundError(
            f"Pricing file not found at: {target_path}. "
            f"Please ensure prices.yaml exists in the repository root."
        )

    with open(target_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    models_data = data.get("models", {})

    def _sync(sess: Session) -> None:
        for info in models_data.values():
            if not isinstance(info, dict):
                continue
            model_id = info.get("model_id")
            if not model_id or str(model_id).strip().startswith("PLACEHOLDER"):
                continue

            model_clean = str(model_id).strip()

            existing = sess.exec(
                select(ModelPrice).where(
                    or_(
                        col(ModelPrice.model) == model_clean,
                        func.lower(col(ModelPrice.model)) == model_clean.lower(),
                    )
                )
            ).first()

            input_rate = info.get("input_per_mtok")
            output_rate = info.get("output_per_mtok")
            source = info.get("source", "Nebius Token Factory")

            if existing:
                existing.model = model_clean
                existing.input_per_mtok = input_rate
                existing.output_per_mtok = output_rate
                existing.source = source
                sess.add(existing)
            else:
                price_record = ModelPrice(
                    model=model_clean,
                    input_per_mtok=input_rate,
                    output_per_mtok=output_rate,
                    source=source,
                )
                sess.add(price_record)
        sess.commit()

    if session is not None:
        _sync(session)
    else:
        with Session(engine) as sess:
            _sync(sess)


def check_live_pricing_verified(yaml_path: str | Path | None = None) -> tuple[bool, str]:
    """Check if prices.yaml has been explicitly verified for live inference.

    Returns (is_verified, reason).
    Permits live execution only when verified_for_live is strictly boolean True.
    """
    target_path = Path(yaml_path) if yaml_path else settings.prices_path
    if not target_path.is_absolute():
        target_path = REPO_ROOT / target_path

    if not target_path.exists():
        return False, f"Pricing file not found at: {target_path}"

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as exc:  # noqa: BLE001
        return False, f"Failed to read pricing file: {exc}"

    if not isinstance(data, dict):
        return False, f"Invalid pricing file format at: {target_path} (expected YAML mapping)"

    if data.get("verified_for_live") is not True:
        return (
            False,
            (
                "prices.yaml has verified_for_live=false (or unset/not boolean True). "
                "Model IDs and pricing rates must be reviewed against official Nebius rates before live dispatch."
            ),
        )
    return True, "Pricing verified for live use."


def resolve_model_price(model_or_alias: str, session: Session) -> tuple[str, ModelPrice]:
    """Resolve real model ID and its pricing record, looking up by model ID first."""
    create_db_and_tables()

    if not model_or_alias or not str(model_or_alias).strip():
        raise ValueError("Model or alias identifier cannot be empty.")

    clean_id = str(model_or_alias).strip()
    m_lower = clean_id.lower()

    # 1. Direct exact match by model ID in database
    price = session.exec(select(ModelPrice).where(ModelPrice.model == clean_id)).first()
    if price:
        return price.model, price

    # 2. Case-insensitive lookup by model ID in database
    price = session.exec(select(ModelPrice).where(func.lower(ModelPrice.model) == m_lower)).first()
    if price:
        return price.model, price

    # 3. Check if model_or_alias is an alias or canonical ID in prices.yaml
    target_path = settings.prices_path
    if not target_path.is_absolute():
        target_path = REPO_ROOT / target_path

    if target_path.exists():
        with open(target_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        models_data = data.get("models", {})

        matched_real_id: str | None = None
        for alias, info in models_data.items():
            if not isinstance(info, dict):
                continue
            alias_clean = str(alias).strip()
            real_id = str(info.get("model_id", "")).strip()
            short_alias = alias_clean.lower().removeprefix("nemotron_").strip()
            if m_lower in (alias_clean.lower(), short_alias, real_id.lower()):
                matched_real_id = real_id
                break

        if matched_real_id:
            if matched_real_id.startswith("PLACEHOLDER"):
                raise ValueError(
                    f"Model '{clean_id}' maps to placeholder ID '{matched_real_id}'. "
                    f"Please update prices.yaml with real model IDs."
                )

            # Query database for canonical model ID
            price = session.exec(
                select(ModelPrice).where(
                    or_(
                        col(ModelPrice.model) == matched_real_id,
                        func.lower(col(ModelPrice.model)) == matched_real_id.lower(),
                    )
                )
            ).first()

            if not price:
                # Self-heal: sync yaml prices into session and retry lookup
                sync_prices_from_yaml(yaml_path=target_path, session=session)
                price = session.exec(
                    select(ModelPrice).where(
                        or_(
                            col(ModelPrice.model) == matched_real_id,
                            func.lower(col(ModelPrice.model)) == matched_real_id.lower(),
                        )
                    )
                ).first()

            if price:
                return price.model, price

    raise ValueError(
        f"Model or alias '{clean_id}' not found in pricing records. "
        f"Ensure prices.yaml has an entry for it with official rates."
    )


def calculate_cost(
    model: str,
    tokens_in: int,
    tokens_out: int,
    session: Session,
) -> float:
    """Calculate cost in USD for a model call or raise ValueError if price is null."""
    _real_id, price = resolve_model_price(model, session)

    if price.input_per_mtok is None or price.output_per_mtok is None:
        raise ValueError(
            f"Pricing for model '{price.model}' has unconfigured (null) values in prices.yaml. "
            f"Official input and output prices per million tokens must be set before computing cost."
        )

    input_cost = (tokens_in * price.input_per_mtok) / 1_000_000.0
    output_cost = (tokens_out * price.output_per_mtok) / 1_000_000.0
    return float(input_cost + output_cost)

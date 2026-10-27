"""Phase 3: Model routing recommendations based on objective and semantic quality."""

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from sqlmodel import Session, col, select

from server.app import db as db_mod
from server.app.config import REPO_ROOT, settings
from server.app.db import create_db_and_tables, resolve_model_price
from server.app.judge import _objective_checks_have_run
from server.app.models import Evaluation, Replay, Run, Step


class TierSummary(BaseModel):
    """Evaluation summary for a specific candidate tier on a tag."""

    tier: str
    model: str
    samples: int
    total_steps: int
    objective_pass_rate: float
    judge_pass_rate: float
    avg_judge_score: float | None
    cost_usd: float
    baseline_cost_usd: float
    qualified: bool
    rejection_reason: str | None = None


class TagRecommendation(BaseModel):
    """Routing recommendation for a step tag."""

    tag: str
    recommended_tier: str
    recommended_model: str
    baseline_tier: str
    baseline_model: str
    samples: int
    objective_pass_rate: float | None = None
    judge_pass_rate: float | None = None
    avg_judge_score: float | None = None
    baseline_cost_usd: float
    recommended_cost_usd: float
    estimated_savings_usd: float
    estimated_savings_pct: float
    decision_reason: str
    tier_evaluations: list[TierSummary] = Field(default_factory=list)


class RunRoutingReport(BaseModel):
    """Complete routing recommendation report for a run."""

    run_id: str
    min_pass_rate: float
    min_samples: int
    baseline_total_cost_usd: float
    recommended_total_cost_usd: float
    estimated_savings_usd: float
    estimated_savings_pct: float
    recommendations: dict[str, TagRecommendation]
    routing_map: dict[str, str]
    disclaimer: str = (
        "Disclaimer: These savings are estimated from isolated step-level replays. "
        "Full end-to-end multi-turn compound savings require closed-loop agent validation."
    )


def _check_model_pricing_configured(model_name: str, session: Session) -> bool:
    """Verify that a model has non-null pricing in the database."""
    try:
        _real_id, price = resolve_model_price(model_name, session)
        return price.input_per_mtok is not None and price.output_per_mtok is not None
    except Exception:  # noqa: BLE001
        return False


def resolve_model_tier_name(model_or_alias: str, session: Session | None = None) -> str:
    """Resolve a model name or alias to its tier identifier or fallback to model name."""
    if not model_or_alias:
        return "unknown"

    m_clean = model_or_alias.strip()
    m_lower = m_clean.lower()

    # 1. Direct tier alias match
    if m_lower in ("ultra", "nemotron_ultra"):
        return "ultra"
    if m_lower in ("super", "nemotron_super"):
        return "super"
    if m_lower in ("nano", "nemotron_nano"):
        return "nano"

    # 2. Check settings environment model IDs
    if settings.ultra_model_id and m_lower == settings.ultra_model_id.lower().strip():
        return "ultra"
    if settings.super_model_id and m_lower == settings.super_model_id.lower().strip():
        return "super"
    if settings.nano_model_id and m_lower == settings.nano_model_id.lower().strip():
        return "nano"

    # 3. Check prices.yaml mappings
    target_path = settings.prices_path
    if not target_path.is_absolute():
        target_path = REPO_ROOT / target_path
    if target_path.exists():
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            models_data = data.get("models", {})
            for alias, info in models_data.items():
                alias_clean = str(alias).strip()
                real_id = str(info.get("model_id", "")).strip()
                if m_clean in (alias_clean, real_id) or m_lower in (
                    alias_clean.lower(),
                    real_id.lower(),
                ):
                    tier_candidate = alias_clean.lower().removeprefix("nemotron_").strip()
                    if tier_candidate:
                        return tier_candidate
        except Exception:  # noqa: BLE001, S110
            pass

    # 4. Keyword heuristics for standard tiers
    if "nano" in m_lower:
        return "nano"
    if "super" in m_lower:
        return "super"
    if "ultra" in m_lower:
        return "ultra"

    # 5. Fallback to model name/alias
    return m_clean


def _canonical_model_id(model_or_alias: str, session: Session) -> str:
    """Resolve a model name or alias to its canonical model ID.

    Handles configured aliases (e.g. nemotron_ultra, ultra), environment
    settings overrides, whitespace, and case-insensitivity without mutating
    unknown model identifiers into misleading values.
    """
    if not model_or_alias:
        return ""
    m_clean = model_or_alias.strip()
    if not m_clean:
        return ""
    m_lower = m_clean.lower()

    # 1. Environment settings overrides
    if (
        m_lower in ("ultra", "nemotron_ultra")
        and settings.ultra_model_id
        and not settings.ultra_model_id.startswith("PLACEHOLDER")
    ):
        return settings.ultra_model_id.strip()
    if (
        m_lower in ("super", "nemotron_super")
        and settings.super_model_id
        and not settings.super_model_id.startswith("PLACEHOLDER")
    ):
        return settings.super_model_id.strip()
    if (
        m_lower in ("nano", "nemotron_nano")
        and settings.nano_model_id
        and not settings.nano_model_id.startswith("PLACEHOLDER")
    ):
        return settings.nano_model_id.strip()

    # 2. Check prices.yaml mappings
    target_path = settings.prices_path
    if not target_path.is_absolute():
        target_path = REPO_ROOT / target_path
    if target_path.exists():
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            models_data = data.get("models", {})
            for alias, info in models_data.items():
                alias_clean = str(alias).strip()
                real_id = str(info.get("model_id", "")).strip()
                short_alias = alias_clean.lower().removeprefix("nemotron_").strip()
                if (
                    m_lower in (alias_clean.lower(), short_alias, real_id.lower())
                    and real_id
                    and not real_id.startswith("PLACEHOLDER")
                ):
                    return real_id
        except Exception:  # noqa: BLE001, S110
            pass

    # 3. Check database pricing lookup
    try:
        real_id, _ = resolve_model_price(m_clean, session)
        return real_id
    except Exception:  # noqa: BLE001
        return m_clean


def _is_replay_evaluated(ev: Evaluation | None) -> bool:
    """Check if an Evaluation exists, objective checks have run, and judge has evaluated."""
    if ev is None:
        return False
    if not _objective_checks_have_run(ev):
        return False
    return bool(ev.judge_reason and ev.judge_reason.strip())


def recommend_routing(
    run_id: str,
    min_pass_rate: float = 0.95,
    min_samples: int = 3,
    session: Session | None = None,
) -> RunRoutingReport:
    """Recommend the cheapest qualifying model tier per step tag.

    Enforces 100% objective checks as a hard constraint and min_pass_rate
    for semantic judge checks over complete matching step coverage.
    """
    if not (0.0 <= min_pass_rate <= 1.0):
        raise ValueError(f"--min-pass-rate must be between 0.0 and 1.0, got {min_pass_rate}.")
    if min_samples < 1:
        raise ValueError(f"--min-samples must be at least 1, got {min_samples}.")

    create_db_and_tables()
    close_session = False
    if session is None:
        session = Session(db_mod.engine)
        close_session = True

    try:
        run_record = session.get(Run, run_id)
        if not run_record:
            raise ValueError(f"Run '{run_id}' not found in database.")

        steps = session.exec(
            select(Step).where(Step.run_id == run_id).order_by(col(Step.idx))
        ).all()

        # Group steps by tag
        tag_steps_map: dict[str, list[Step]] = {}
        for s in steps:
            tag_steps_map.setdefault(s.tag, []).append(s)

        recommendations: dict[str, TagRecommendation] = {}
        routing_map: dict[str, str] = {}
        total_baseline_cost = 0.0
        total_recommended_cost = 0.0

        for tag, tag_steps in tag_steps_map.items():
            step_count = len(tag_steps)
            step_ids = {s.id for s in tag_steps}
            raw_baseline_model = run_record.baseline_model or tag_steps[0].model
            baseline_model = _canonical_model_id(raw_baseline_model, session)
            baseline_tier = resolve_model_tier_name(baseline_model, session)
            tag_baseline_cost = sum(s.cost_usd for s in tag_steps)
            total_baseline_cost += tag_baseline_cost

            # Collect all replays for these steps
            replays = session.exec(select(Replay).where(col(Replay.step_id).in_(step_ids))).all()

            tier_replays_map: dict[str, list[Replay]] = {}
            for r in replays:
                tier_replays_map.setdefault(r.tier, []).append(r)

            tier_summaries: list[TierSummary] = []
            qualifying_tiers: list[TierSummary] = []

            # Evaluate each candidate tier
            for tier, t_replays in tier_replays_map.items():
                raw_tier_model = t_replays[0].model if t_replays else tier
                tier_model = _canonical_model_id(raw_tier_model, session)
                pricing_ok = _check_model_pricing_configured(tier_model, session)

                obj_passed_count = 0
                judge_passed_count = 0
                scores: list[float] = []
                tier_cost = sum(r.cost_usd for r in t_replays)

                evaluated_replays: list[Replay] = []
                unevaluated_replays: list[Replay] = []
                error_replays = [r for r in t_replays if r.error is not None]

                for r in t_replays:
                    if r.error is not None:
                        # Replay error is treated as failure
                        continue

                    ev = session.exec(
                        select(Evaluation).where(Evaluation.replay_id == r.id)
                    ).first()
                    if not _is_replay_evaluated(ev):
                        unevaluated_replays.append(r)
                        continue

                    assert ev is not None
                    evaluated_replays.append(r)

                    # Objective check: must have actually run and passed
                    if ev.objective_pass:
                        obj_passed_count += 1

                    # Semantic judge check: only [PASS] counts
                    if ev.judge_reason and ev.judge_reason.startswith("[PASS]"):
                        judge_passed_count += 1

                    if ev.judge_score is not None:
                        scores.append(ev.judge_score)

                evaluated_samples = len(evaluated_replays)

                # Rates are calculated strictly over the total steps for this tag
                obj_pass_rate = obj_passed_count / step_count if step_count > 0 else 0.0
                judge_pass_rate = judge_passed_count / step_count if step_count > 0 else 0.0
                avg_score = (sum(scores) / len(scores)) if scores else None

                replayed_step_ids = [r.step_id for r in t_replays]
                unique_replayed_step_ids = set(replayed_step_ids)
                missing_step_ids = step_ids - unique_replayed_step_ids
                extraneous_step_ids = unique_replayed_step_ids - step_ids
                has_duplicates = len(replayed_step_ids) != len(unique_replayed_step_ids)

                is_baseline = (
                    (tier_model.lower() == baseline_model.lower())
                    or tier.lower().strip() == baseline_tier.lower().strip()
                    or resolve_model_tier_name(tier_model, session).lower() == baseline_tier.lower()
                )

                # Determine qualification with strict coverage & validity checks
                qualified = True
                rejection_reason: str | None = None

                if is_baseline:
                    qualified = False
                    rejection_reason = "Baseline model cannot be recommended as its own downgrade"
                elif step_count < min_samples:
                    qualified = False
                    rejection_reason = f"Insufficient tag samples ({step_count} < {min_samples})"
                elif missing_step_ids:
                    qualified = False
                    rejection_reason = f"Incomplete tier coverage (missing replay for {len(missing_step_ids)}/{step_count} steps)"
                elif extraneous_step_ids:
                    qualified = False
                    rejection_reason = f"Extraneous replays detected for step(s) outside tag ({len(extraneous_step_ids)} steps)"
                elif has_duplicates:
                    qualified = False
                    rejection_reason = f"Duplicate replays detected ({len(t_replays)} replays for {len(unique_replayed_step_ids)} steps)"
                elif error_replays:
                    qualified = False
                    rejection_reason = f"Replay error on {len(error_replays)}/{step_count} step(s): {error_replays[0].error}"
                elif unevaluated_replays:
                    qualified = False
                    rejection_reason = f"Missing or incomplete evaluation for {len(unevaluated_replays)}/{step_count} replay(s)"
                elif evaluated_samples < min_samples:
                    qualified = False
                    rejection_reason = f"Insufficient evaluated candidate samples ({evaluated_samples} < {min_samples})"
                elif not pricing_ok:
                    qualified = False
                    rejection_reason = f"Pricing unconfigured for model '{tier_model}'"
                elif obj_pass_rate < 1.0:
                    qualified = False
                    rejection_reason = (
                        f"Objective failure ({obj_passed_count}/{step_count} passed, required 100%)"
                    )
                elif judge_pass_rate < min_pass_rate:
                    qualified = False
                    rejection_reason = (
                        f"Judge pass rate {judge_pass_rate * 100:.1f}% < {min_pass_rate * 100:.1f}% "
                        f"({judge_passed_count}/{step_count} passed)"
                    )

                summary = TierSummary(
                    tier=tier,
                    model=tier_model,
                    samples=evaluated_samples,
                    total_steps=step_count,
                    objective_pass_rate=obj_pass_rate,
                    judge_pass_rate=judge_pass_rate,
                    avg_judge_score=avg_score,
                    cost_usd=tier_cost,
                    baseline_cost_usd=tag_baseline_cost,
                    qualified=qualified,
                    rejection_reason=rejection_reason,
                )
                tier_summaries.append(summary)
                if qualified and tier_cost < tag_baseline_cost:
                    qualifying_tiers.append(summary)

            # Pick winning tier: lowest cost among qualified cheaper tiers
            if qualifying_tiers:
                qualifying_tiers.sort(key=lambda t: t.cost_usd)
                winner = qualifying_tiers[0]
                rec_tier = winner.tier
                rec_model = _canonical_model_id(winner.model, session)
                rec_cost = winner.cost_usd
                savings_usd = max(0.0, tag_baseline_cost - rec_cost)
                savings_pct = (
                    (savings_usd / tag_baseline_cost * 100.0) if tag_baseline_cost > 0 else 0.0
                )
                decision_reason = (
                    f"Selected cheapest qualifying tier with 100% objective and "
                    f"{winner.judge_pass_rate * 100:.1f}% judge pass rate"
                )
                rec_obj_rate = winner.objective_pass_rate
                rec_judge_rate = winner.judge_pass_rate
                rec_avg_score = winner.avg_judge_score
            else:
                # Retain baseline: represent unmeasured baseline metrics honestly as None
                rec_tier = baseline_tier
                rec_model = baseline_model
                rec_cost = tag_baseline_cost
                savings_usd = 0.0
                savings_pct = 0.0
                rec_obj_rate = None
                rec_judge_rate = None
                rec_avg_score = None
                if step_count < min_samples:
                    decision_reason = (
                        f"Insufficient samples ({step_count} < {min_samples}); retained baseline"
                    )
                elif tier_summaries:
                    reasons_str = "; ".join(
                        f"{ts.tier}: {ts.rejection_reason}"
                        for ts in tier_summaries
                        if ts.rejection_reason
                    )
                    decision_reason = (
                        f"No cheaper tier qualified ({reasons_str}); retained baseline"
                    )
                else:
                    decision_reason = "No candidate replays evaluated; retained baseline"

            tag_rec = TagRecommendation(
                tag=tag,
                recommended_tier=rec_tier,
                recommended_model=rec_model,
                baseline_tier=baseline_tier,
                baseline_model=baseline_model,
                samples=step_count,
                objective_pass_rate=rec_obj_rate,
                judge_pass_rate=rec_judge_rate,
                avg_judge_score=rec_avg_score,
                baseline_cost_usd=tag_baseline_cost,
                recommended_cost_usd=rec_cost,
                estimated_savings_usd=savings_usd,
                estimated_savings_pct=savings_pct,
                decision_reason=decision_reason,
                tier_evaluations=tier_summaries,
            )
            recommendations[tag] = tag_rec
            routing_map[tag] = rec_model
            total_recommended_cost += rec_cost

        total_savings_usd = max(0.0, total_baseline_cost - total_recommended_cost)
        total_savings_pct = (
            (total_savings_usd / total_baseline_cost * 100.0) if total_baseline_cost > 0 else 0.0
        )

        return RunRoutingReport(
            run_id=run_id,
            min_pass_rate=min_pass_rate,
            min_samples=min_samples,
            baseline_total_cost_usd=total_baseline_cost,
            recommended_total_cost_usd=total_recommended_cost,
            estimated_savings_usd=total_savings_usd,
            estimated_savings_pct=total_savings_pct,
            recommendations=recommendations,
            routing_map=routing_map,
        )
    finally:
        if close_session:
            session.close()


def export_routing_config(report: RunRoutingReport, export_path: str | Path) -> None:
    """Export the tag-to-model routing map as JSON to the specified path."""
    target_path = Path(export_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(report.routing_map, f, indent=2)


def load_routing_config(
    routing_source: str | Path | dict[str, Any] | None,
    session: Session | None = None,
) -> dict[str, str]:
    """Load and validate a routing configuration.

    Parameters:
        routing_source: Path to JSON file, parsed dict, or None.
        session: Optional SQLModel session for model pricing validation.

    Returns:
        A dict mapping tag names to model IDs. If routing_source is None, returns {}.

    Raises:
        FileNotFoundError: If the routing file does not exist.
        ValueError: If JSON is malformed, root is not a dict, keys/values are invalid,
                    or any model cannot be resolved to configured pricing.
    """
    if routing_source is None:
        return {}

    raw_data: Any
    if isinstance(routing_source, (str, Path)):
        target_path = Path(routing_source)
        if not target_path.exists():
            raise FileNotFoundError(f"Routing config file not found: {target_path}")
        if not target_path.is_file():
            raise ValueError(f"Routing config path is not a file: {target_path}")
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in routing config file '{target_path}': {exc}") from exc
    elif isinstance(routing_source, dict):
        raw_data = routing_source
    else:
        raise TypeError(
            f"Routing configuration must be a file path or dict, got {type(routing_source).__name__}."
        )

    if not isinstance(raw_data, dict):
        raise TypeError("Routing configuration must be a JSON object mapping tags to model IDs.")

    validated_map: dict[str, str] = {}
    close_session = False
    if session is None:
        create_db_and_tables()
        try:
            db_mod.sync_prices_from_yaml()
        except (FileNotFoundError, ValueError):
            pass
        session = Session(db_mod.engine)
        close_session = True

    try:
        for tag, model in raw_data.items():
            if not isinstance(tag, str):
                raise TypeError("Routing config tags must be strings.")
            if not tag.strip():
                raise ValueError("Routing config tags must be non-empty strings.")
            tag_clean = tag.strip()

            if not isinstance(model, str):
                raise TypeError(f"Model ID for tag '{tag_clean}' must be a string.")
            if not model.strip():
                raise ValueError(f"Model ID for tag '{tag_clean}' must be a non-empty string.")
            model_clean = model.strip()

            try:
                _real_model_id, price = resolve_model_price(model_clean, session)
            except Exception as exc:
                raise ValueError(
                    f"Unknown or unpriced model ID '{model_clean}' for tag '{tag_clean}': {exc}"
                ) from exc

            if price.input_per_mtok is None or price.output_per_mtok is None:
                raise ValueError(
                    f"Model '{model_clean}' for tag '{tag_clean}' has incomplete (null) pricing rates."
                )

            validated_map[tag_clean] = model_clean

        return validated_map
    finally:
        if close_session:
            session.close()

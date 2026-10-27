"""User-defined research question and evaluation checklist for Downshift benchmark.

Checklist items are groups of accepted keywords matched with regex word boundaries
(case-insensitive, any one keyword in a group matches).
"""

import re
from collections.abc import Sequence
from typing import Any

ChecklistGroup = Sequence[str] | str

# User-defined research question and evaluation checklist for Downshift:
RESEARCH_QUESTION: str = (
    "What NVIDIA Nemotron models are hosted on Nebius Token Factory, "
    "what are their architecture tiers and specifications, "
    "and what agent workloads are they designed for?"
)
CHECKLIST_ITEMS: list[ChecklistGroup] = [
    ["nebius", "token factory"],
    ["nemotron", "nemotron-3", "nemotron-4"],
    ["ultra", "super", "nano", "340b", "70b", "120b", "30b", "4b"],
    ["function calling", "tool use", "reasoning", "agent", "rag"],
]


def evaluate_checklist_compliance(
    final_answer: str, checklist: Sequence[ChecklistGroup] | None = None
) -> dict[str, Any]:
    """Check whether the final answer contains any keyword from each checklist group."""
    items = checklist if checklist is not None else CHECKLIST_ITEMS
    if not items:
        return {
            "all_passed": False,
            "passed_count": 0,
            "total_count": 0,
            "checklist_score": 0.0,
            "found_groups": [],
            "missing_groups": [],
            "details": {},
        }

    details = {}
    found_groups = []
    missing_groups = []
    answer_lower = final_answer.lower()

    for idx, item in enumerate(items):
        keywords = [item] if isinstance(item, str) else list(item)
        matched_kw = next(
            (
                kw
                for kw in keywords
                if re.search(r"\b" + re.escape(kw.lower()) + r"\b", answer_lower)
            ),
            None,
        )
        group_repr = "/".join(keywords)

        if matched_kw is not None:
            details[group_repr] = {"matched": True, "keyword": matched_kw}
            found_groups.append(keywords)
        else:
            details[group_repr] = {"matched": False, "keyword": None}
            missing_groups.append(keywords)

    score = len(found_groups) / float(len(items))
    all_passed = len(found_groups) == len(items)

    return {
        "all_passed": all_passed,
        "passed_count": len(found_groups),
        "total_count": len(items),
        "checklist_score": score,
        "found_groups": found_groups,
        "missing_groups": missing_groups,
        "details": details,
    }

# mosaicx/pipelines/rewards.py
"""Reward functions for dspy.Refine and dspy.BestOfN inside extraction pipelines.

These score a single prediction's quality (no gold label needed).
Separate from evaluation/rewards.py which compares against gold labels.
"""

from __future__ import annotations

import re
from typing import Any

_MEASUREMENT_RE = re.compile(r"\d+\s*(mm|cm|m|cc|ml|mg|g|kg)", re.IGNORECASE)


def findings_reward(findings: list[dict[str, Any]]) -> float:
    """Score extraction quality of a findings list."""
    if not findings:
        return 0.0

    score = 0.3  # non-empty findings
    anatomy_bonus = sum(
        0.1 for f in findings
        if f.get("anatomy") and str(f["anatomy"]).strip()
    )
    score += min(anatomy_bonus, 0.4)

    measure_bonus = sum(
        0.1 for f in findings
        if _MEASUREMENT_RE.search(str(f.get("description", "")))
    )
    score += min(measure_bonus, 0.2)

    observation_bonus = sum(
        0.05 for f in findings
        if f.get("observation") and str(f["observation"]).strip()
    )
    score += min(observation_bonus, 0.1)

    return min(score, 1.0)


def impression_reward(impressions: list[dict[str, Any]]) -> float:
    """Score extraction quality of an impressions list."""
    if not impressions:
        return 0.0

    score = 0.3  # non-empty
    for imp in impressions:
        stmt = str(imp.get("statement", ""))
        if len(stmt) > 10:
            score += 0.2
        if imp.get("actionable"):
            score += 0.1
    return min(score, 1.0)


def diagnosis_reward(diagnoses: list[dict[str, Any]]) -> float:
    """Score extraction quality of a diagnosis list."""
    if not diagnoses:
        return 0.0

    score = 0.3
    for dx in diagnoses:
        if dx.get("diagnosis") and str(dx["diagnosis"]).strip():
            score += 0.2
        if dx.get("grade"):
            score += 0.1
        if dx.get("margin"):
            score += 0.1
    return min(score, 1.0)


def schema_compliance_reward(
    extraction: dict[str, Any],
    required_fields: list[str],
) -> float:
    """Score how well an extraction matches the expected schema fields."""
    if not required_fields:
        return 1.0

    present = sum(
        1 for f in required_fields
        if f in extraction and extraction[f] is not None and str(extraction[f]).strip()
    )
    return present / len(required_fields)

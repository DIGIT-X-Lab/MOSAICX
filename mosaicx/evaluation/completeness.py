# mosaicx/evaluation/completeness.py
"""Completeness evaluator for structured medical reports.

Provides field coverage analysis, information density measurement,
and an overall completeness score combining both metrics.

Key functions:
    - field_coverage: Fraction of non-empty fields in a Pydantic model.
    - information_density: Token ratio of structured output vs source text.
    - compute_completeness: Weighted combination of coverage and density.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel


def _is_populated(value: Any) -> bool:
    """Return True if the value counts as 'populated'.

    A value is considered populated if it is:
        - not None
        - not an empty string
        - not an empty list/dict/set
    """
    if value is None:
        return False
    if isinstance(value, str) and value.strip() == "":
        return False
    if isinstance(value, (list, dict, set)) and len(value) == 0:
        return False
    return True


def field_coverage(model: BaseModel) -> float:
    """Compute the fraction of populated fields in a Pydantic model.

    Iterates over every field defined on *model*. Each field scores 1.0 if
    its current value is considered populated (non-None, non-empty string,
    non-empty collection), and 0.0 otherwise.

    Parameters
    ----------
    model:
        A Pydantic BaseModel instance whose fields will be inspected.

    Returns
    -------
    float
        Mean coverage score in [0.0, 1.0].  Returns 0.0 if the model has
        no fields.
    """
    fields = model.model_fields
    if not fields:
        return 0.0

    scores: list[float] = []
    for name in fields:
        value = getattr(model, name)
        scores.append(1.0 if _is_populated(value) else 0.0)

    return sum(scores) / len(scores)


def information_density(source: str, structured: BaseModel) -> float:
    """Compute the token-level information density ratio.

    Converts the structured model to a JSON string, then measures the
    ratio of whitespace-delimited tokens in the structured output to
    those in the source text.  The result is clamped to [0.0, 1.0].

    Parameters
    ----------
    source:
        The original unstructured text from which *structured* was derived.
    structured:
        A Pydantic BaseModel representing the structured extraction.

    Returns
    -------
    float
        Density ratio in [0.0, 1.0].  Returns 0.0 if *source* is empty.
    """
    if not source or not source.strip():
        return 0.0

    structured_json = structured.model_dump_json()
    source_tokens = source.split()
    structured_tokens = structured_json.split()

    if len(source_tokens) == 0:
        return 0.0

    ratio = len(structured_tokens) / len(source_tokens)
    return min(ratio, 1.0)


def compute_completeness(
    structured: BaseModel,
    source_text: str,
    coverage_weight: float = 0.7,
    density_weight: float = 0.3,
) -> dict[str, float]:
    """Compute an overall completeness score for a structured report.

    Combines :func:`field_coverage` and :func:`information_density` into
    a single weighted average.

    Parameters
    ----------
    structured:
        The structured report as a Pydantic model.
    source_text:
        The original unstructured source text.
    coverage_weight:
        Weight for the field coverage component (default 0.7).
    density_weight:
        Weight for the information density component (default 0.3).

    Returns
    -------
    dict[str, float]
        Dictionary with keys ``"overall"``, ``"field_coverage"``, and
        ``"information_density"``, each a float in [0.0, 1.0].
    """
    fc = field_coverage(structured)
    density = information_density(source_text, structured)
    overall = coverage_weight * fc + density_weight * density

    return {
        "overall": overall,
        "field_coverage": fc,
        "information_density": density,
    }

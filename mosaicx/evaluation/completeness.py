# mosaicx/evaluation/completeness.py
"""Completeness evaluator for structured medical reports.

Provides field coverage analysis, information density measurement,
and an overall completeness score combining both metrics.

Key functions:
    - field_coverage: Fraction of non-empty fields in a Pydantic model.
    - information_density: Token ratio of structured output vs source text.
    - compute_completeness: Weighted combination of coverage and density.
    - compute_report_completeness: Template-aware per-field scoring.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
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


# ---------------------------------------------------------------------------
# Template-aware report completeness
# ---------------------------------------------------------------------------


@dataclass
class FieldCompleteness:
    """Per-field completeness status."""

    name: str
    filled: bool
    required: bool
    value_summary: str  # truncated repr or "---"


@dataclass
class ReportCompleteness:
    """Full completeness breakdown for a structured report."""

    overall: float  # weighted score [0, 1]
    field_coverage: float  # unweighted fraction of filled fields
    required_coverage: float  # fraction of required fields filled
    optional_coverage: float  # fraction of optional fields filled
    information_density: float  # token ratio
    fields: list[FieldCompleteness] = field(default_factory=list)
    total_fields: int = 0
    filled_fields: int = 0
    missing_required: list[str] = field(default_factory=list)


def _value_summary(value: Any, max_len: int = 60) -> str:
    """Return a short string summary of a field value."""
    if not _is_populated(value):
        return "---"
    text = repr(value)
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def compute_report_completeness(
    model_instance: BaseModel,
    source_text: str,
    model_class: type[BaseModel] | None = None,
    required_weight: float = 1.0,
    optional_weight: float = 0.3,
) -> ReportCompleteness:
    """Compute template-aware completeness with per-field breakdown.

    Parameters
    ----------
    model_instance:
        A populated Pydantic model instance to score.
    source_text:
        Original unstructured source text (for information density).
    model_class:
        The Pydantic model class to inspect for required/optional metadata.
        Defaults to ``type(model_instance)``.
    required_weight:
        Weight given to required fields in the overall score.
    optional_weight:
        Weight given to optional fields in the overall score.

    Returns
    -------
    ReportCompleteness
        Detailed completeness breakdown.
    """
    if model_class is None:
        model_class = type(model_instance)

    model_fields = model_class.model_fields
    if not model_fields:
        return ReportCompleteness(
            overall=0.0,
            field_coverage=0.0,
            required_coverage=0.0,
            optional_coverage=0.0,
            information_density=0.0,
        )

    per_field: list[FieldCompleteness] = []
    required_filled = 0
    required_total = 0
    optional_filled = 0
    optional_total = 0
    total_weight = 0.0
    weighted_score = 0.0

    for name, field_info in model_fields.items():
        value = getattr(model_instance, name, None)
        filled = _is_populated(value)
        is_required = field_info.is_required()

        per_field.append(
            FieldCompleteness(
                name=name,
                filled=filled,
                required=is_required,
                value_summary=_value_summary(value),
            )
        )

        if is_required:
            required_total += 1
            weight = required_weight
            if filled:
                required_filled += 1
        else:
            optional_total += 1
            weight = optional_weight
            if filled:
                optional_filled += 1

        total_weight += weight
        if filled:
            weighted_score += weight

    total_filled = required_filled + optional_filled
    total_fields = required_total + optional_total
    fc = total_filled / total_fields if total_fields else 0.0
    req_cov = required_filled / required_total if required_total else 1.0
    opt_cov = optional_filled / optional_total if optional_total else 1.0
    overall = weighted_score / total_weight if total_weight else 0.0
    density = information_density(source_text, model_instance)

    missing_req = [
        f.name for f in per_field if f.required and not f.filled
    ]

    return ReportCompleteness(
        overall=overall,
        field_coverage=fc,
        required_coverage=req_cov,
        optional_coverage=opt_cov,
        information_density=density,
        fields=per_field,
        total_fields=total_fields,
        filled_fields=total_filled,
        missing_required=missing_req,
    )

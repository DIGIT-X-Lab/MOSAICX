"""Quality-gate tests for query/verify grounding and numeric correctness."""

from __future__ import annotations

from types import SimpleNamespace


def test_quality_gate_query_numeric_grounded_passes_threshold():
    from mosaicx.evaluation.grounding import query_grounded_numeric_metric

    example = SimpleNamespace(
        question="What is the average BMI?",
        response="24.5",
        expected_numeric=24.5,
    )
    prediction = SimpleNamespace(
        response="The average BMI is 24.5.",
        context="Computed mean BMI from cohort table: 24.5.",
    )

    score = query_grounded_numeric_metric(example, prediction)
    assert score >= 0.8


def test_quality_gate_query_numeric_mismatch_fails_threshold():
    from mosaicx.evaluation.grounding import query_grounded_numeric_metric

    example = SimpleNamespace(
        question="What is the average BMI?",
        response="24.5",
        expected_numeric=24.5,
    )
    prediction = SimpleNamespace(
        response="The average BMI is 31.2.",
        context="Computed mean BMI from cohort table: 31.2.",
    )

    score = query_grounded_numeric_metric(example, prediction)
    assert score < 0.5


def test_quality_gate_verify_verdict_consistency_thresholds():
    from mosaicx.evaluation.grounding import verify_grounded_metric

    gold = SimpleNamespace(verdict="verified")

    good = SimpleNamespace(verdict="verified", confidence=0.95)
    bad = SimpleNamespace(verdict="contradicted", confidence=0.95)

    assert verify_grounded_metric(gold, good) >= 0.9
    assert verify_grounded_metric(gold, bad) <= 0.2

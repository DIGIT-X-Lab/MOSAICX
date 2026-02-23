from __future__ import annotations

from types import SimpleNamespace


def test_numeric_exactness_metric_exact_match():
    from mosaicx.evaluation.grounding import numeric_exactness_metric

    example = SimpleNamespace(expected_numeric=25)
    prediction = SimpleNamespace(response="Mean BMI is 25")
    score = numeric_exactness_metric(example, prediction)
    assert score == 1.0


def test_numeric_exactness_metric_penalizes_mismatch():
    from mosaicx.evaluation.grounding import numeric_exactness_metric

    example = SimpleNamespace(expected_numeric=25)
    prediction = SimpleNamespace(response="Mean BMI is 30")
    score = numeric_exactness_metric(example, prediction)
    assert score < 0.5


def test_complete_and_grounded_metric_heuristic_path():
    from mosaicx.evaluation.grounding import complete_and_grounded_metric

    example = SimpleNamespace(question="What modality was used?", response="CT")
    prediction = SimpleNamespace(response="CT", context="Imaging report states: Modality: CT")
    score = complete_and_grounded_metric(example, prediction)
    assert 0.0 <= score <= 1.0
    assert score >= 0.5


def test_query_grounded_numeric_metric_combines_grounding_and_numeric():
    from mosaicx.evaluation.grounding import query_grounded_numeric_metric

    example = SimpleNamespace(question="What is average BMI?", response="25", expected_numeric=25)
    prediction = SimpleNamespace(response="Average BMI is 25", context="Computed mean BMI: 25")
    score = query_grounded_numeric_metric(example, prediction)
    assert score >= 0.8


def test_verify_grounded_metric_rewards_matching_verdict():
    from mosaicx.evaluation.grounding import verify_grounded_metric

    example = SimpleNamespace(verdict="verified")
    prediction = SimpleNamespace(verdict="verified", confidence=0.9)
    score = verify_grounded_metric(example, prediction)
    assert score >= 0.9

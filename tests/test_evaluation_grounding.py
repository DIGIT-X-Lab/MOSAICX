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


def test_query_metric_components_expose_grounding_and_numeric_parts():
    from mosaicx.evaluation.grounding import query_metric_components

    example = SimpleNamespace(question="Mean BMI?", response="25", expected_numeric=25)
    prediction = SimpleNamespace(response="Mean BMI is 25", context="Computed mean BMI: 25")
    parts = query_metric_components(example, prediction)
    assert parts["has_numeric_target"] is True
    assert 0.0 <= parts["grounding"] <= 1.0
    assert 0.0 <= parts["numeric"] <= 1.0
    assert 0.0 <= parts["exact_match"] <= 1.0
    assert 0.0 <= parts["passage_match"] <= 1.0
    assert 0.0 <= parts["score"] <= 1.0


def test_verify_metric_components_expose_verdict_match_and_confidence():
    from mosaicx.evaluation.grounding import verify_metric_components

    example = SimpleNamespace(verdict="verified")
    prediction = SimpleNamespace(verdict="contradicted", confidence=0.9)
    parts = verify_metric_components(example, prediction)
    assert parts["verdict_match"] == 0.0
    assert 0.0 <= parts["confidence"] <= 1.0
    assert 0.0 <= parts["score"] <= 1.0


def test_answer_exact_match_metric_fallback_path():
    from mosaicx.evaluation.grounding import answer_exact_match_metric

    example = SimpleNamespace(question="Modality?", response="CT")
    prediction = SimpleNamespace(response="CT")
    assert answer_exact_match_metric(example, prediction) == 1.0


def test_answer_passage_match_metric_fallback_path():
    from mosaicx.evaluation.grounding import answer_passage_match_metric

    example = SimpleNamespace(question="Modality?", response="CT")
    prediction = SimpleNamespace(
        response="CT",
        context="Imaging report states: Modality CT was used throughout.",
    )
    score = answer_passage_match_metric(example, prediction)
    assert 0.0 <= score <= 1.0
    assert score > 0.0

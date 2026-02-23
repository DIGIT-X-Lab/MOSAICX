from __future__ import annotations


def test_query_quality_gates_pass_for_grounded_numeric_rows():
    from mosaicx.evaluation.quality_gates import evaluate_query_quality_gates

    rows = [
        {"grounding": 0.91, "numeric": 1.0, "has_numeric_target": True, "score": 0.964},
        {"grounding": 0.88, "numeric": 0.98, "has_numeric_target": True, "score": 0.94},
        {"grounding": 0.84, "numeric": 1.0, "has_numeric_target": False, "score": 0.84},
    ]
    report = evaluate_query_quality_gates(rows)
    assert report["passed"] is True
    assert report["summary"]["mean_score"] >= 0.75
    assert report["summary"]["grounding_mean"] >= 0.70


def test_query_quality_gates_fail_on_numeric_regression():
    from mosaicx.evaluation.quality_gates import evaluate_query_quality_gates

    rows = [
        {"grounding": 0.92, "numeric": 0.20, "has_numeric_target": True, "score": 0.488},
        {"grounding": 0.89, "numeric": 0.30, "has_numeric_target": True, "score": 0.536},
    ]
    report = evaluate_query_quality_gates(rows)
    assert report["passed"] is False
    failed_names = {c["name"] for c in report["failed_checks"]}
    assert "numeric_mean" in failed_names or "numeric_pass_rate" in failed_names


def test_verify_quality_gates_pass_for_high_verdict_accuracy():
    from mosaicx.evaluation.quality_gates import evaluate_verify_quality_gates

    rows = [
        {"verdict_match": 1.0, "confidence": 0.95, "score": 0.99},
        {"verdict_match": 1.0, "confidence": 0.85, "score": 0.97},
        {"verdict_match": 1.0, "confidence": 0.80, "score": 0.96},
    ]
    report = evaluate_verify_quality_gates(rows)
    assert report["passed"] is True
    assert report["summary"]["verdict_accuracy"] >= 0.90


def test_verify_quality_gates_fail_on_verdict_mismatch():
    from mosaicx.evaluation.quality_gates import evaluate_verify_quality_gates

    rows = [
        {"verdict_match": 0.0, "confidence": 0.95, "score": 0.19},
        {"verdict_match": 1.0, "confidence": 0.90, "score": 0.98},
    ]
    report = evaluate_verify_quality_gates(rows)
    assert report["passed"] is False
    failed_names = {c["name"] for c in report["failed_checks"]}
    assert "verdict_accuracy" in failed_names


def test_evaluate_quality_gates_rejects_unsupported_pipeline():
    import pytest

    from mosaicx.evaluation.quality_gates import evaluate_quality_gates

    with pytest.raises(ValueError):
        evaluate_quality_gates("radiology", [])


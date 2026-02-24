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


def test_query_quality_gates_include_exact_and_passage_checks_when_present():
    from mosaicx.evaluation.quality_gates import evaluate_query_quality_gates

    rows = [
        {
            "grounding": 0.93,
            "numeric": 1.0,
            "has_numeric_target": True,
            "exact_match": 1.0,
            "passage_match": 0.9,
            "score": 0.972,
        },
        {
            "grounding": 0.88,
            "numeric": 0.98,
            "has_numeric_target": True,
            "exact_match": 0.8,
            "passage_match": 0.8,
            "score": 0.932,
        },
    ]
    report = evaluate_query_quality_gates(rows)
    check_names = {c["name"] for c in report["checks"]}
    assert "exact_match_mean" in check_names
    assert "passage_match_mean" in check_names
    assert report["passed"] is True


def test_query_quality_gates_fail_on_exact_or_passage_regression():
    from mosaicx.evaluation.quality_gates import evaluate_query_quality_gates

    rows = [
        {
            "grounding": 0.92,
            "numeric": 1.0,
            "has_numeric_target": True,
            "exact_match": 0.0,
            "passage_match": 0.1,
            "score": 0.968,
        },
        {
            "grounding": 0.90,
            "numeric": 0.99,
            "has_numeric_target": True,
            "exact_match": 0.0,
            "passage_match": 0.2,
            "score": 0.954,
        },
    ]
    report = evaluate_query_quality_gates(rows)
    assert report["passed"] is False
    failed_names = {c["name"] for c in report["failed_checks"]}
    assert "exact_match_mean" in failed_names or "passage_match_mean" in failed_names

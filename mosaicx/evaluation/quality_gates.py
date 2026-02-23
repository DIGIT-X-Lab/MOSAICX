"""Hard quality-gate checks for query/verify evaluation and optimization."""

from __future__ import annotations

from typing import Any


DEFAULT_QUERY_GATES: dict[str, float] = {
    "min_mean_score": 0.75,
    "min_grounding_mean": 0.70,
    "min_numeric_mean": 0.90,
    "min_numeric_pass_rate": 0.85,
    "numeric_pass_threshold": 0.95,
}

DEFAULT_VERIFY_GATES: dict[str, float] = {
    "min_mean_score": 0.90,
    "min_verdict_accuracy": 0.90,
    "min_confidence_mean": 0.70,
}


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _build_check(name: str, value: float, threshold: float) -> dict[str, Any]:
    passed = float(value) >= float(threshold)
    return {
        "name": name,
        "value": float(value),
        "threshold": float(threshold),
        "passed": bool(passed),
    }


def evaluate_query_quality_gates(
    rows: list[dict[str, Any]],
    *,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    cfg = dict(DEFAULT_QUERY_GATES)
    if thresholds:
        cfg.update(thresholds)

    scores = [float(r.get("score", 0.0) or 0.0) for r in rows]
    grounding = [float(r.get("grounding", 0.0) or 0.0) for r in rows]
    numeric_rows = [r for r in rows if bool(r.get("has_numeric_target"))]
    numeric_scores = [float(r.get("numeric", 0.0) or 0.0) for r in numeric_rows]

    summary: dict[str, Any] = {
        "count": len(rows),
        "mean_score": _mean(scores),
        "grounding_mean": _mean(grounding),
        "numeric_count": len(numeric_rows),
        "numeric_mean": _mean(numeric_scores) if numeric_scores else None,
        "numeric_pass_rate": None,
    }
    if numeric_scores:
        pass_threshold = float(cfg["numeric_pass_threshold"])
        pass_rate = sum(1 for s in numeric_scores if s >= pass_threshold) / len(numeric_scores)
        summary["numeric_pass_rate"] = float(pass_rate)

    checks = [
        _build_check("mean_score", summary["mean_score"], float(cfg["min_mean_score"])),
        _build_check("grounding_mean", summary["grounding_mean"], float(cfg["min_grounding_mean"])),
    ]
    if numeric_scores:
        checks.append(
            _build_check("numeric_mean", float(summary["numeric_mean"] or 0.0), float(cfg["min_numeric_mean"]))
        )
        checks.append(
            _build_check(
                "numeric_pass_rate",
                float(summary["numeric_pass_rate"] or 0.0),
                float(cfg["min_numeric_pass_rate"]),
            )
        )

    passed = bool(rows) and all(bool(c["passed"]) for c in checks)
    failed = [c for c in checks if not c["passed"]]
    return {
        "pipeline": "query",
        "passed": passed,
        "summary": summary,
        "checks": checks,
        "failed_checks": failed,
    }


def evaluate_verify_quality_gates(
    rows: list[dict[str, Any]],
    *,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    cfg = dict(DEFAULT_VERIFY_GATES)
    if thresholds:
        cfg.update(thresholds)

    scores = [float(r.get("score", 0.0) or 0.0) for r in rows]
    verdict_matches = [float(r.get("verdict_match", 0.0) or 0.0) for r in rows]
    confidences = [float(r.get("confidence", 0.0) or 0.0) for r in rows]

    summary = {
        "count": len(rows),
        "mean_score": _mean(scores),
        "verdict_accuracy": _mean(verdict_matches),
        "confidence_mean": _mean(confidences),
    }
    checks = [
        _build_check("mean_score", summary["mean_score"], float(cfg["min_mean_score"])),
        _build_check("verdict_accuracy", summary["verdict_accuracy"], float(cfg["min_verdict_accuracy"])),
        _build_check("confidence_mean", summary["confidence_mean"], float(cfg["min_confidence_mean"])),
    ]
    passed = bool(rows) and all(bool(c["passed"]) for c in checks)
    failed = [c for c in checks if not c["passed"]]
    return {
        "pipeline": "verify",
        "passed": passed,
        "summary": summary,
        "checks": checks,
        "failed_checks": failed,
    }


def evaluate_quality_gates(
    pipeline: str,
    rows: list[dict[str, Any]],
    *,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    name = str(pipeline).strip().lower()
    if name == "query":
        return evaluate_query_quality_gates(rows, thresholds=thresholds)
    if name == "verify":
        return evaluate_verify_quality_gates(rows, thresholds=thresholds)
    raise ValueError("Quality gates are currently supported only for 'query' and 'verify'.")


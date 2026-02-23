"""Grounding and numeric-exactness evaluation helpers for query/verify workflows."""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "on", "to", "for", "with",
    "is", "are", "was", "were", "be", "been", "being", "what", "which",
    "who", "whom", "when", "where", "why", "how", "does", "do", "did",
}


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _tokens(text: str) -> set[str]:
    return {
        t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= 2 and t not in _STOPWORDS
    }


def _extract_context(prediction: Any) -> str:
    if hasattr(prediction, "context"):
        value = _as_text(getattr(prediction, "context"))
        if value:
            return value

    citations = getattr(prediction, "citations", None)
    if isinstance(citations, list):
        snippets = [
            " ".join(_as_text(c.get("snippet", "")).split())
            for c in citations
            if isinstance(c, dict)
        ]
        return "\n".join(s for s in snippets if s)
    return ""


def _heuristic_grounded_score(question: str, response: str, context: str) -> float:
    context_terms = _tokens(context)
    if not context_terms:
        return 0.0

    q_terms = _tokens(question)
    r_terms = _tokens(response)

    question_support = (
        len(q_terms & context_terms) / len(q_terms) if q_terms else 1.0
    )
    response_support = (
        len(r_terms & context_terms) / len(r_terms) if r_terms else 1.0
    )
    return max(0.0, min(1.0, (0.45 * question_support) + (0.55 * response_support)))


def numeric_exactness_metric(
    example: Any,
    prediction: Any,
    trace: Any = None,
) -> float:
    """Score numeric exactness for query/verify answers.

    Accepts either:
    - ``example.expected_numeric`` / ``prediction.response``
    - or free-text ``example.response`` / ``prediction.response``
      (first numeric literal extracted from each)
    """
    expected_raw = getattr(example, "expected_numeric", None)
    if expected_raw is None:
        expected_raw = getattr(example, "response", "")
    predicted_raw = getattr(prediction, "response", None)
    if predicted_raw is None:
        predicted_raw = getattr(prediction, "answer", "")

    expected_match = _NUM_RE.search(_as_text(expected_raw))
    if expected_match is None:
        # No numeric target in gold label.
        return 1.0

    predicted_match = _NUM_RE.search(_as_text(predicted_raw))
    if predicted_match is None:
        return 0.0

    try:
        expected_val = float(expected_match.group(0))
        predicted_val = float(predicted_match.group(0))
    except ValueError:
        return 0.0

    abs_err = abs(predicted_val - expected_val)
    tol = max(1e-9, abs(expected_val) * 0.01)
    if abs_err <= tol:
        return 1.0

    # Smooth decay beyond tolerance.
    rel_err = abs_err / max(abs(expected_val), 1.0)
    score = max(0.0, 1.0 - (rel_err * 5.0))
    return max(0.0, min(1.0, score))


def complete_and_grounded_metric(
    example: Any,
    prediction: Any,
    trace: Any = None,
) -> float:
    """Score answer completeness+groundedness.

    Uses ``dspy.evaluate.CompleteAndGrounded`` when available and configured.
    Falls back to a deterministic lexical groundedness heuristic otherwise.
    """
    question = _as_text(getattr(example, "question", ""))
    reference = _as_text(getattr(example, "response", ""))
    response = _as_text(getattr(prediction, "response", getattr(prediction, "answer", "")))
    context = _extract_context(prediction)

    try:
        import dspy

        if getattr(dspy.settings, "lm", None) is not None and hasattr(dspy, "evaluate"):
            complete_cls = getattr(dspy.evaluate, "CompleteAndGrounded", None)
            if complete_cls is not None:
                metric = complete_cls()
                ex = SimpleNamespace(question=question, response=reference)
                pred = SimpleNamespace(response=response, context=context)
                value = metric(ex, pred, trace=trace)
                return float(value)
    except Exception:
        pass

    return _heuristic_grounded_score(question, response, context)


def query_grounded_numeric_metric(
    example: Any,
    prediction: Any,
    trace: Any = None,
) -> float:
    """Combined metric for query outputs: grounding + numeric exactness."""
    grounding = complete_and_grounded_metric(example, prediction, trace=trace)

    expected_numeric = getattr(example, "expected_numeric", None)
    has_numeric_target = expected_numeric is not None or _NUM_RE.search(
        _as_text(getattr(example, "response", ""))
    )
    if not has_numeric_target:
        return grounding

    numeric = numeric_exactness_metric(example, prediction, trace=trace)
    return (0.4 * grounding) + (0.6 * numeric)


def verify_grounded_metric(
    example: Any,
    prediction: Any,
    trace: Any = None,
) -> float:
    """Metric for verify outputs: verdict correctness + grounding confidence."""
    gold_verdict = _as_text(getattr(example, "verdict", getattr(example, "response", ""))).strip().lower()
    pred_verdict = _as_text(getattr(prediction, "verdict", getattr(prediction, "response", ""))).strip().lower()

    verdict_score = 1.0 if gold_verdict and gold_verdict == pred_verdict else 0.0

    pred_conf = getattr(prediction, "confidence", None)
    try:
        conf_score = float(pred_conf) if pred_conf is not None else 0.5
    except (TypeError, ValueError):
        conf_score = 0.5
    conf_score = max(0.0, min(1.0, conf_score))

    return (0.8 * verdict_score) + (0.2 * conf_score)

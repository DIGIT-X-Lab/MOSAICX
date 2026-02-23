"""DSPy metric functions for each MOSAICX pipeline.

Each metric follows the DSPy convention::

    metric(example, prediction, trace=None) -> float   # [0.0, 1.0]

``example`` carries both inputs and gold-standard labels (set via
``.with_inputs()``).  ``prediction`` is the DSPy module output.

Registry helpers:
    - ``get_metric(pipeline)`` — returns the metric function
    - ``list_metrics()``       — returns available pipeline names
"""

from __future__ import annotations

from typing import Any, Callable

from .grounding import query_grounded_numeric_metric, verify_grounded_metric


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_list(val: Any) -> list:
    """Coerce *val* to a list (handles None, str, dict)."""
    if val is None:
        return []
    if isinstance(val, str):
        return [val]
    if isinstance(val, dict):
        return [val]
    try:
        return list(val)
    except TypeError:
        return [val]


def _safe_str(val: Any) -> str:
    """Coerce *val* to str."""
    if val is None:
        return ""
    return str(val)


def _safe_dict(val: Any) -> dict:
    """Coerce *val* to dict."""
    if isinstance(val, dict):
        return val
    return {}


def _token_set(text: str) -> set[str]:
    """Lowercase token set for overlap scoring."""
    return set(text.lower().split())


# ---------------------------------------------------------------------------
# Radiology metric
# ---------------------------------------------------------------------------

def radiology_metric(example: Any, prediction: Any, trace: Any = None) -> float:
    """Score radiology extraction quality.

    Components:
    - extraction_reward on predicted findings/impressions (weight 0.6)
    - exam_type exact match with gold (weight 0.2)
    - finding count similarity (weight 0.2)
    """
    from .rewards import extraction_reward

    # --- extraction reward (0.6) ---
    pred_findings = _safe_list(getattr(prediction, "findings", []))
    pred_impression = _safe_str(getattr(prediction, "impressions", ""))

    # Convert findings to dicts for extraction_reward
    finding_dicts = []
    for f in pred_findings:
        if isinstance(f, dict):
            finding_dicts.append(f)
        elif hasattr(f, "__dict__"):
            finding_dicts.append(vars(f))
        else:
            finding_dicts.append({"description": str(f)})

    ext_score = extraction_reward(finding_dicts, pred_impression)

    # --- exam_type match (0.2) ---
    gold_exam = _safe_str(getattr(example, "exam_type", "")).strip().lower()
    pred_exam = _safe_str(getattr(prediction, "exam_type", "")).strip().lower()
    exam_score = 1.0 if (gold_exam and gold_exam == pred_exam) else 0.0

    # --- finding count similarity (0.2) ---
    gold_findings = _safe_list(getattr(example, "findings", []))
    gold_count = len(gold_findings)
    pred_count = len(pred_findings)
    if gold_count == 0 and pred_count == 0:
        count_score = 1.0
    elif gold_count == 0 or pred_count == 0:
        count_score = 0.0
    else:
        count_score = min(gold_count, pred_count) / max(gold_count, pred_count)

    return 0.6 * ext_score + 0.2 * exam_score + 0.2 * count_score


# ---------------------------------------------------------------------------
# Pathology metric
# ---------------------------------------------------------------------------

def pathology_metric(example: Any, prediction: Any, trace: Any = None) -> float:
    """Score pathology extraction quality.

    Components:
    - finding quality: fraction of predicted findings with non-empty
      description (weight 0.4)
    - diagnosis count similarity vs gold (weight 0.3)
    - specimen type exact match (weight 0.3)
    """
    # --- finding quality (0.4) ---
    pred_findings = _safe_list(getattr(prediction, "findings", []))
    if pred_findings:
        described = sum(
            1 for f in pred_findings
            if _safe_str(f.get("description", "") if isinstance(f, dict) else getattr(f, "description", ""))
        )
        finding_score = described / len(pred_findings)
    else:
        finding_score = 0.0

    # --- diagnosis count (0.3) ---
    gold_findings = _safe_list(getattr(example, "findings", []))
    gold_count = len(gold_findings)
    pred_count = len(pred_findings)
    if gold_count == 0 and pred_count == 0:
        diag_score = 1.0
    elif gold_count == 0 or pred_count == 0:
        diag_score = 0.0
    else:
        diag_score = min(gold_count, pred_count) / max(gold_count, pred_count)

    # --- specimen type match (0.3) ---
    gold_spec = _safe_str(getattr(example, "specimen_type", "")).strip().lower()
    pred_spec = _safe_str(getattr(prediction, "specimen_type", "")).strip().lower()
    spec_score = 1.0 if (gold_spec and gold_spec == pred_spec) else 0.0

    return 0.4 * finding_score + 0.3 * diag_score + 0.3 * spec_score


# ---------------------------------------------------------------------------
# Extraction metric
# ---------------------------------------------------------------------------

def extraction_metric(example: Any, prediction: Any, trace: Any = None) -> float:
    """Score generic document extraction quality.

    Compares predicted ``extracted`` dict against gold ``extracted`` dict:
    - key overlap (weight 0.5)
    - value exact match for shared keys (weight 0.5)
    """
    gold = _safe_dict(getattr(example, "extracted", {}))
    pred = _safe_dict(getattr(prediction, "extracted", {}))

    if not gold:
        # No gold labels — score based on non-emptiness
        return 1.0 if pred else 0.0

    gold_keys = set(gold.keys())
    pred_keys = set(pred.keys())

    # --- key overlap (0.5) ---
    if gold_keys:
        key_score = len(gold_keys & pred_keys) / len(gold_keys)
    else:
        key_score = 1.0

    # --- value match for shared keys (0.5) ---
    shared = gold_keys & pred_keys
    if shared:
        matches = sum(
            1 for k in shared
            if _safe_str(gold[k]).strip().lower() == _safe_str(pred[k]).strip().lower()
        )
        val_score = matches / len(shared)
    else:
        val_score = 0.0

    return 0.5 * key_score + 0.5 * val_score


# ---------------------------------------------------------------------------
# Summarizer metric
# ---------------------------------------------------------------------------

def summarizer_metric(example: Any, prediction: Any, trace: Any = None) -> float:
    """Score timeline summarization quality.

    Components:
    - narrative length adequacy (weight 0.4): at least 50 chars
    - keyword overlap with gold narrative (weight 0.6)
    """
    pred_narrative = _safe_str(getattr(prediction, "narrative", ""))
    gold_narrative = _safe_str(getattr(example, "narrative", ""))

    # --- length adequacy (0.4) ---
    length_score = min(len(pred_narrative) / 50.0, 1.0) if pred_narrative else 0.0

    # --- keyword overlap (0.6) ---
    if gold_narrative:
        gold_tokens = _token_set(gold_narrative)
        pred_tokens = _token_set(pred_narrative)
        if gold_tokens:
            overlap_score = len(gold_tokens & pred_tokens) / len(gold_tokens)
        else:
            overlap_score = 1.0
    else:
        # No gold narrative — score based on non-emptiness
        overlap_score = 1.0 if pred_narrative.strip() else 0.0

    return 0.4 * length_score + 0.6 * overlap_score


# ---------------------------------------------------------------------------
# Deidentifier metric
# ---------------------------------------------------------------------------

def deidentifier_metric(example: Any, prediction: Any, trace: Any = None) -> float:
    """Score de-identification quality.

    Components:
    - PHI leak safety score (weight 0.6)
    - token overlap with gold redacted text (weight 0.4)
    """
    from .rewards import phi_leak_reward

    pred_text = _safe_str(getattr(prediction, "redacted_text", ""))

    # --- PHI leak score (0.6) ---
    phi_score = phi_leak_reward(pred_text)

    # --- gold text overlap (0.4) ---
    gold_text = _safe_str(getattr(example, "redacted_text", ""))
    if gold_text:
        gold_tokens = _token_set(gold_text)
        pred_tokens = _token_set(pred_text)
        if gold_tokens:
            overlap_score = len(gold_tokens & pred_tokens) / len(gold_tokens)
        else:
            overlap_score = 1.0
    else:
        overlap_score = 1.0 if pred_text.strip() else 0.0

    return 0.6 * phi_score + 0.4 * overlap_score


# ---------------------------------------------------------------------------
# Schema generation metric
# ---------------------------------------------------------------------------

def schema_gen_metric(example: Any, prediction: Any, trace: Any = None) -> float:
    """Score schema generation quality.

    Compares predicted SchemaSpec field names against gold field names:
    - field name overlap (weight 0.7)
    - penalty for extra fields not in gold (weight 0.3)
    """
    # Extract gold field names
    gold_spec = getattr(example, "schema_spec", None)
    if gold_spec is None:
        gold_fields_raw = _safe_list(getattr(example, "fields", []))
    elif isinstance(gold_spec, dict):
        gold_fields_raw = gold_spec.get("fields", [])
    elif hasattr(gold_spec, "fields"):
        gold_fields_raw = gold_spec.fields
    else:
        gold_fields_raw = []

    gold_names = set()
    for f in gold_fields_raw:
        if isinstance(f, dict):
            gold_names.add(f.get("name", "").lower())
        elif hasattr(f, "name"):
            gold_names.add(f.name.lower())

    gold_names.discard("")

    # Extract predicted field names
    pred_spec = getattr(prediction, "schema_spec", None)
    if pred_spec is None:
        return 0.0
    if isinstance(pred_spec, dict):
        pred_fields_raw = pred_spec.get("fields", [])
    elif hasattr(pred_spec, "fields"):
        pred_fields_raw = pred_spec.fields
    else:
        pred_fields_raw = []

    pred_names = set()
    for f in pred_fields_raw:
        if isinstance(f, dict):
            pred_names.add(f.get("name", "").lower())
        elif hasattr(f, "name"):
            pred_names.add(f.name.lower())

    pred_names.discard("")

    if not gold_names:
        return 1.0 if not pred_names else 0.5

    # --- overlap (0.7) ---
    overlap = len(gold_names & pred_names) / len(gold_names)

    # --- extra penalty (0.3) ---
    extra = pred_names - gold_names
    extra_penalty = len(extra) / max(len(gold_names), 1)
    extra_score = max(0.0, 1.0 - extra_penalty)

    return 0.7 * overlap + 0.3 * extra_score


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_METRIC_REGISTRY: dict[str, Callable] = {
    "radiology": radiology_metric,
    "pathology": pathology_metric,
    "extract": extraction_metric,
    "summarize": summarizer_metric,
    "deidentify": deidentifier_metric,
    "schema": schema_gen_metric,
    "query": query_grounded_numeric_metric,
    "verify": verify_grounded_metric,
}


def get_metric(pipeline: str) -> Callable:
    """Return the metric function for *pipeline*.

    Raises
    ------
    ValueError
        If *pipeline* is not recognised.
    """
    if pipeline not in _METRIC_REGISTRY:
        raise ValueError(
            f"No metric for pipeline '{pipeline}'. "
            f"Available: {sorted(_METRIC_REGISTRY)}"
        )
    return _METRIC_REGISTRY[pipeline]


def list_metrics() -> list[str]:
    """Return sorted list of pipeline names with metrics."""
    return sorted(_METRIC_REGISTRY)
